// @vitest-environment jsdom
import { afterEach, describe, expect, it } from 'vitest';
import { act, cleanup, render, screen, within } from '@testing-library/react';
import { MockSim } from '../src/net/mockServer';
import { AckToast } from '../src/ui/AckToast';
import { setAudioContextForTest } from '../src/ui/chime';
import { WarningMonitor, blindSpots, loudest, overSpeedLimit, useWarningEffects } from '../src/ui/warnings';
import { PARAM_DEFS, useSimStore } from '../src/store/simStore';
import type { StateUpdate } from '../src/schema';
import { createHarness, resetStore, type Harness } from './harness';

const BASE = new MockSim().frame();
const DEFAULTS = Object.fromEntries(PARAM_DEFS.map((d) => [d.key, d.default]));

interface Over {
  t?: number;
  speed?: number;
  brake?: number;
  ttc?: number | null;
  mode?: StateUpdate['ego']['cruise']['mode'];
  fsdAvailable?: boolean;
  limit?: number;
  maneuver?: StateUpdate['plan']['maneuver'];
  target?: number;
}

function frame(o: Over = {}): StateUpdate {
  return {
    ...BASE,
    t: o.t ?? 1,
    events: [],
    detections: [],
    ego: {
      ...BASE.ego,
      speed_mps: o.speed ?? 10,
      brake: o.brake ?? 0,
      speed_limit_mps: o.limit ?? 11.18,
      cruise: {
        ...BASE.ego.cruise,
        mode: o.mode ?? 'off',
        fsd_available: o.fsdAvailable ?? true,
      },
    },
    plan: { ...BASE.plan, maneuver: o.maneuver ?? 'keep_lane', target_speed_mps: o.target ?? 11 },
    telemetry: { ...BASE.telemetry, ttc_s: o.ttc ?? null },
  };
}

const codes = (m: WarningMonitor, f: StateUpdate, params = DEFAULTS) => m.update(f, params).map((w) => w.code);

describe('WarningMonitor', () => {
  it('warns of a forward collision once, on the rising edge', () => {
    const m = new WarningMonitor();
    expect(codes(m, frame({ t: 1, ttc: 4 }))).toEqual([]);
    expect(codes(m, frame({ t: 1.1, ttc: 2.3 }))).toEqual(['fcw']);
    expect(codes(m, frame({ t: 1.2, ttc: 2.0 }))).toEqual([]); // still on: no repeat
    expect(codes(m, frame({ t: 1.3, ttc: 5 }))).toEqual([]);
    expect(codes(m, frame({ t: 1.4, ttc: 2.0 }))).toEqual(['fcw']); // a new episode
  });

  it('keeps FCW quiet while braking, at a crawl, when off, and later on Late', () => {
    expect(codes(new WarningMonitor(), frame({ ttc: 2.3, brake: 0.5 }))).toEqual([]);
    expect(codes(new WarningMonitor(), frame({ ttc: 2.3, speed: 1 }))).toEqual([]);
    expect(codes(new WarningMonitor(), frame({ ttc: 2.3 }), { ...DEFAULTS, fcw_sensitivity: 'off' }))
      .toEqual([]);
    expect(codes(new WarningMonitor(), frame({ ttc: 2.3 }), { ...DEFAULTS, fcw_sensitivity: 'late' }))
      .toEqual([]);
    expect(codes(new WarningMonitor(), frame({ ttc: 2.8 }), { ...DEFAULTS, fcw_sensitivity: 'early' }))
      .toEqual(['fcw']);
  });

  it("announces the planner's emergency braking where the planner drives", () => {
    expect(codes(new WarningMonitor(), frame({ maneuver: 'emergency_brake', mode: 'fsd', brake: 1 })))
      .toEqual(['aeb']);
    expect(codes(new WarningMonitor(), frame({ maneuver: 'emergency_brake', mode: 'cruise', brake: 1 })))
      .toEqual(['aeb']);
    // Off-route Cruise holds speed, and Manual says so with a backend event.
    expect(codes(new WarningMonitor(), frame({ maneuver: 'emergency_brake', mode: 'cruise', fsdAvailable: false })))
      .toEqual([]);
    expect(codes(new WarningMonitor(), frame({ maneuver: 'emergency_brake', mode: 'off' }))).toEqual([]);
  });

  it('warns a driver closing on a stop too fast, not one the car is stopping for', () => {
    const fast = frame({ maneuver: 'stop', target: 4, speed: 9 });
    expect(codes(new WarningMonitor(), fast)).toEqual(['stop_warning']);
    expect(codes(new WarningMonitor(), { ...fast, ego: { ...fast.ego, brake: 0.6 } })).toEqual([]);
    expect(codes(new WarningMonitor(), frame({ maneuver: 'stop', target: 4, speed: 9, mode: 'fsd' })))
      .toEqual([]);
    expect(codes(new WarningMonitor(), frame({ maneuver: 'stop', target: 4, speed: 6 }))).toEqual([]);
  });

  it('chimes once when the stop being waited at releases, only if asked to', () => {
    const on = { ...DEFAULTS, green_light_chime: 'on' };
    const m = new WarningMonitor();
    codes(m, frame({ t: 1, maneuver: 'stop', speed: 0, target: 0 }), on);
    expect(codes(m, frame({ t: 2, maneuver: 'keep_lane', speed: 0 }), on)).toEqual(['green_light']);
    expect(codes(m, frame({ t: 3, maneuver: 'keep_lane', speed: 0 }), on)).toEqual([]);
    const quiet = new WarningMonitor();
    codes(quiet, frame({ t: 1, maneuver: 'stop', speed: 0, target: 0 }));
    expect(codes(quiet, frame({ t: 2, maneuver: 'keep_lane', speed: 0 }))).toEqual([]);
  });

  it('measures Speed Assist relative to the limit, or against a fixed speed', () => {
    // Limit 25 mph (11.18 m/s), offset 5: 31 mph is over, 29 is not.
    expect(overSpeedLimit(frame({ speed: 13.9 }), DEFAULTS)).toBe(true);
    expect(overSpeedLimit(frame({ speed: 13.0 }), DEFAULTS)).toBe(false);
    expect(overSpeedLimit(frame({ speed: 13.9, limit: 0 }), DEFAULTS)).toBe(false);
    const abs = { ...DEFAULTS, speed_limit_mode: 'absolute', speed_limit_absolute_mph: 20 };
    expect(overSpeedLimit(frame({ speed: 9.5 }), abs)).toBe(true);
    expect(overSpeedLimit(frame({ speed: 13.9 }), { ...DEFAULTS, speed_limit_warning: 'off' })).toBe(false);
  });

  it('chimes for speed only in Chime mode and only after a second over', () => {
    const chimeMode = { ...DEFAULTS, speed_limit_warning: 'chime' };
    const m = new WarningMonitor();
    expect(codes(m, frame({ t: 1, speed: 14 }), chimeMode)).toEqual([]);
    expect(codes(m, frame({ t: 2.1, speed: 14 }), chimeMode)).toEqual(['speed_limit']);
    const display = new WarningMonitor();
    codes(display, frame({ t: 1, speed: 14 }));
    expect(codes(display, frame({ t: 3, speed: 14 }))).toEqual([]);
  });

  it('asks for a takeover when Assist or FSD loses its route', () => {
    expect(codes(new WarningMonitor(), frame({ mode: 'fsd', fsdAvailable: false }))).toEqual(['take_over']);
    expect(codes(new WarningMonitor(), frame({ mode: 'off', fsdAvailable: false }))).toEqual([]);
  });

  it('warns of an unsignalled lane departure in Manual and Cruise, once per drift', () => {
    const lane = (offset: number, left: 'dashed_white' | 'none' = 'dashed_white') => ({
      ...BASE.telemetry.lane,
      lane_width_m: 3.6,
      offset_m: offset,
      left_marking: left,
      right_marking: 'solid_white' as const,
    });
    const at = (t: number, offset: number, extra: Partial<StateUpdate['ego']> = {}, left?: 'none') => {
      const f = frame({ t, speed: 8 });
      return {
        ...f,
        ego: { ...f.ego, size: { ...f.ego.size, width: 1.9 }, ...extra },
        telemetry: { ...f.telemetry, lane: lane(offset, left) },
      };
    };
    const m = new WarningMonitor();
    expect(codes(m, at(1, 0.5))).toEqual([]);
    expect(codes(m, at(1.1, 0.9))).toEqual(['lane_departure']); // edge = 1.8 - 0.95 = 0.85
    expect(codes(m, at(1.2, 0.7))).toEqual([]); // inside the hysteresis: still the same drift
    expect(codes(m, at(1.3, 0.4))).toEqual([]); // back in lane
    expect(codes(m, at(1.4, 0.9))).toEqual(['lane_departure']); // a new drift
    // Signalled, unmarked, engaged, or slow: quiet.
    expect(codes(new WarningMonitor(), at(1, 0.9, { turn_signal: 'left' }))).toEqual([]);
    expect(codes(new WarningMonitor(), at(1, 0.9, {}, 'none'))).toEqual([]);
    const fsd = at(1, -0.9);
    expect(codes(new WarningMonitor(), { ...fsd, ego: { ...fsd.ego, cruise: { ...fsd.ego.cruise, mode: 'fsd' } } }))
      .toEqual([]);
    expect(codes(new WarningMonitor(), at(1, 0.9, { speed_mps: 3 }))).toEqual([]);
    expect(codes(new WarningMonitor(), at(1, 0.9), { ...DEFAULTS, lane_departure_avoidance: 'off' })).toEqual([]);
  });

  it('finds a moving car in a blind spot, not one ahead, far behind, or parked', () => {
    // Ego heading north (+y) at the origin: its left is -x.
    const withAgent = (x: number, y: number, speed = 8, egoSpeed = 10) => {
      const f = frame({ speed: egoSpeed });
      return {
        ...f,
        ego: { ...f.ego, pose: { x: 0, y: 0, heading: Math.PI / 2 } },
        world_agents: [{ id: 'a', cls: 'car' as const, pose: { x, y, heading: Math.PI / 2 }, size: f.ego.size, speed_mps: speed }],
      };
    };
    expect(blindSpots(withAgent(-3.6, -4))).toEqual({ left: true, right: false });
    expect(blindSpots(withAgent(3.6, -2))).toEqual({ left: false, right: true });
    expect(blindSpots(withAgent(-3.6, -12))).toEqual({ left: false, right: false }); // far behind
    expect(blindSpots(withAgent(-3.6, 6))).toEqual({ left: false, right: false }); // ahead: visible
    expect(blindSpots(withAgent(-3.6, -4, 0))).toEqual({ left: false, right: false }); // parked
    expect(blindSpots(withAgent(-3.6, -4, 8, 1))).toEqual({ left: false, right: false }); // ego crawling
    // The warning itself waits for the indicator to point into it.
    const f = withAgent(-3.6, -4);
    expect(codes(new WarningMonitor(), f)).toEqual([]);
    expect(codes(new WarningMonitor(), { ...f, ego: { ...f.ego, turn_signal: 'left' } })).toEqual(['blind_spot']);
  });

  it('says it is pulling over for an emergency vehicle, or just that one is there', () => {
    expect(codes(new WarningMonitor(), frame({ maneuver: 'pull_over', mode: 'fsd' }))).toEqual(['pull_over']);
    const ev = { ...BASE.detections[0] ?? {}, id: 'ev', emergency: true } as StateUpdate['detections'][number];
    const manual = { ...frame({ mode: 'off' }), detections: [ev] };
    expect(codes(new WarningMonitor(), manual)).toEqual(['emergency_vehicle']);
  });

  it('plays the loudest sound of a batch', () => {
    const m = new WarningMonitor();
    const batch = m.update(frame({ ttc: 2 }), DEFAULTS);
    expect(loudest(batch.map((w) => w.sound))).toBe('critical');
    expect(loudest(['info', 'warn', null])).toBe('warn');
    expect(loudest([])).toBeNull();
  });
});

class FakeAudio {
  state = 'running';
  currentTime = 0;
  destination = {};
  oscillators = 0;
  resume() {
    return Promise.resolve();
  }
  createOscillator() {
    this.oscillators++;
    return { type: '', frequency: { value: 0 }, connect: (g: unknown) => g, start() {}, stop() {} };
  }
  createGain() {
    return {
      gain: { setValueAtTime() {}, linearRampToValueAtTime() {} },
      connect: (d: unknown) => d,
    };
  }
}

function Warnings() {
  useWarningEffects();
  return <AckToast />;
}

describe('useWarningEffects', () => {
  let harness: Harness | null = null;
  afterEach(() => {
    cleanup();
    harness?.detach();
    harness = null;
    resetStore();
    setAudioContextForTest(null);
  });

  it('shows and chimes a forward collision warning where the driver is looking', () => {
    const audio = new FakeAudio();
    setAudioContextForTest(audio as unknown as AudioContext);
    harness = createHarness();
    render(<Warnings />);
    harness.emitScene();
    harness.emit(frame({ t: 1, ttc: 4 }));
    expect(audio.oscillators).toBe(0);
    harness.emit(frame({ t: 1.1, ttc: 2.2 }));
    const status = screen.getByRole('status');
    expect(within(status).getByText('Forward collision')).toBeTruthy();
    const beeps = audio.oscillators;
    expect(beeps).toBeGreaterThan(0);
    harness.emit(frame({ t: 1.2, ttc: 2.0 })); // still on: silent
    expect(audio.oscillators).toBe(beeps);
  });

  it('chimes for an intervention the backend made, without logging it twice', () => {
    const audio = new FakeAudio();
    setAudioContextForTest(audio as unknown as AudioContext);
    harness = createHarness();
    render(<Warnings />);
    harness.emitScene();
    const aeb = { t: 1, level: 'critical' as const, code: 'aeb', message: 'emergency braking for car', progress: null };
    harness.emit({ ...frame({ t: 1 }), events: [aeb] });
    expect(audio.oscillators).toBeGreaterThan(0);
    expect(useSimStore.getState().events.filter((e) => e.code === 'aeb')).toHaveLength(1);
    expect(within(screen.getByRole('status')).getByText('Emergency braking')).toBeTruthy();
  });

  it('stays silent with chimes off', () => {
    const audio = new FakeAudio();
    setAudioContextForTest(audio as unknown as AudioContext);
    harness = createHarness();
    render(<Warnings />);
    harness.emitScene();
    act(() => useSimStore.getState().setParam('chimes', 'off'));
    harness.emit(frame({ t: 1.1, ttc: 2.2 }));
    expect(audio.oscillators).toBe(0);
  });
});
