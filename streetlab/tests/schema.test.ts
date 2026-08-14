import { describe, expect, it } from 'vitest';
import {
  AckSchema,
  CommandSchema,
  PROTOCOL_VERSION,
  SceneDescriptionSchema,
  StateUpdateSchema,
  parseCommand,
  parseServerMessage,
} from '../src/schema';
import type { StateUpdate } from '../src/schema';
import { buildScene } from '../src/net/mockCity';

it('is protocol 2', () => {
  expect(PROTOCOL_VERSION).toBe(2);
});

it('accepts load_location with and without a radius', () => {
  expect(parseCommand({ cmd: 'load_location', id: 'c1', query: 'Nob Hill' }).ok).toBe(true);
  expect(
    parseCommand({ cmd: 'load_location', id: 'c2', query: 'Nob Hill', radius_m: 400 }).ok,
  ).toBe(true);
});

it('rejects an empty load_location query', () => {
  expect(parseCommand({ cmd: 'load_location', id: 'c', query: '' }).ok).toBe(false);
});

it('rejects a non-positive load_location radius', () => {
  expect(
    parseCommand({ cmd: 'load_location', id: 'c', query: 'x', radius_m: 0 }).ok,
  ).toBe(false);
  expect(
    parseCommand({ cmd: 'load_location', id: 'c', query: 'x', radius_m: -5 }).ok,
  ).toBe(false);
});

it('rejects an explicit null radius_m (optional means absent, not null)', () => {
  // The Cycle 1 design doc's warning, pinned as a test: `.optional()` allows
  // the key to be missing but not present-and-null — that would need
  // `.nullable()`. Python's receiving side is deliberately more lenient (see
  // schema.py's LoadLocation), but this schema — the one that guards what we
  // put on the wire — must not emit or accept `null` here.
  const res = parseCommand({ cmd: 'load_location', id: 'c', query: 'x', radius_m: null });
  expect(res.ok).toBe(false);
});

/** A minimal but complete StateUpdate, written by hand rather than generated. */
const sample: StateUpdate = {
  type: 'state_update',
  protocol: PROTOCOL_VERSION,
  seq: 42,
  t: 0.7,
  sim_rate_hz: 60,
  paused: false,
  assist_active: true,
  scenario_id: 'nob-hill-loop',
  ego: {
    pose: { x: 5.4, y: 21.3, heading: Math.PI / 2 },
    speed_mps: 14.3,
    accel_mps2: 0.4,
    steering_angle: -0.02,
    yaw_rate: 0.01,
    throttle: 0.19,
    brake: 0,
    gear: 'D',
    speed_limit_mps: 11.176,
    cruise: { mode: 'fsd', set_speed_mps: 20.1 },
    size: { length: 4.9, width: 1.96, height: 1.44 },
  },
  detections: [
    {
      id: 'veh_cutin',
      cls: 'car',
      pose: { x: 5.1, y: 34.8, heading: Math.PI / 2 },
      size: { length: 4.6, width: 1.9, height: 1.46 },
      velocity: [0.1, 12.2],
      speed_mps: 12.2,
      confidence: 0.97,
      hazard: true,
      hazard_label: 'Cut-in vehicle',
      ttc_s: 2.4,
      lane_offset: 0,
    },
  ],
  plan: {
    polyline: [
      [5.4, 21.3],
      [5.4, 31.3],
      [5.4, 41.3],
    ],
    target_speed_mps: 11.176,
    maneuver: 'keep_lane',
    confidence: 0.94,
  },
  telemetry: {
    radar: [
      {
        id: 'veh_cutin',
        azimuth: 0.02,
        range_m: 13.5,
        range_rate_mps: -2.1,
        rcs_db: 11.2,
        tracked: true,
      },
    ],
    lane: {
      lane_index: 0,
      lane_count: 2,
      lane_width_m: 3.6,
      offset_m: 0.12,
      heading_error: 0.004,
      left_marking: 'double_yellow',
      right_marking: 'dashed_white',
      neighbors: [
        {
          id: 'veh_cutin',
          cls: 'car',
          lane_offset: 0,
          longitudinal_m: 13.5,
          lateral_m: 0.2,
          speed_mps: 12.2,
          hazard: true,
        },
      ],
    },
    ttc_s: 2.4,
    vehicle: {
      battery_pct: 77.4,
      range_km: 356,
      motor_temp_c: 57.7,
      tire_pressure_kpa: [248, 249, 245, 246],
      subsystems: [
        { key: 'planning', label: 'Planning', status: 'ok', detail: null },
      ],
      overall: 'ok',
    },
    trajectory: {
      horizon_s: 5,
      planned: [
        { t: -1, lateral_m: 0.05 },
        { t: 0, lateral_m: 0.12 },
        { t: 2.5, lateral_m: 0.5 },
      ],
      cutin: [
        { t: 0, lateral_m: -3.6 },
        { t: 2.5, lateral_m: -0.8 },
      ],
      cutin_label: 'Cut-in vehicle',
    },
  },
  signals: [{ id: 'tl_0_0_e', phase: 'green', time_to_change_s: 6.2 }],
  events: [
    { t: 0.7, level: 'warn', code: 'CUTIN_DETECTED', message: 'Vehicle cutting in' },
  ],
};

describe('StateUpdate', () => {
  it('round-trips a sample through parse without loss', () => {
    const parsed = StateUpdateSchema.parse(sample);
    expect(parsed).toEqual(sample);

    // And survives a real JSON hop, which is how it will actually arrive.
    const overWire = StateUpdateSchema.parse(JSON.parse(JSON.stringify(sample)));
    expect(overWire).toEqual(sample);
  });

  it('is routed by the server-message envelope', () => {
    const res = parseServerMessage(sample);
    expect(res.ok).toBe(true);
    if (res.ok) expect(res.value.type).toBe('state_update');
  });

  it('rejects a frame with a bad enum and reports the path', () => {
    const bad = { ...sample, ego: { ...sample.ego, gear: 'X' } };
    const res = parseServerMessage(bad);
    expect(res.ok).toBe(false);
    if (!res.ok) expect(res.error).toContain('ego.gear');
  });

  it('rejects out-of-range confidence', () => {
    const bad = structuredClone(sample);
    bad.detections[0].confidence = 1.4;
    expect(StateUpdateSchema.safeParse(bad).success).toBe(false);
  });

  it('strips unknown keys so a newer backend stays compatible', () => {
    const forward = { ...sample, some_future_field: 123 };
    const parsed = StateUpdateSchema.parse(forward);
    expect('some_future_field' in parsed).toBe(false);
  });
});

describe('SceneDescription', () => {
  it('validates the hand-authored mock city', () => {
    const scene = buildScene('nob-hill-loop');
    expect(() => SceneDescriptionSchema.parse(scene)).not.toThrow();
    expect(parseServerMessage(scene).ok).toBe(true);
  });

  it('rejects a building footprint with fewer than three points', () => {
    const scene = buildScene('nob-hill-loop');
    const bad = structuredClone(scene);
    bad.buildings[0].footprint = [[0, 0], [1, 1]];
    expect(SceneDescriptionSchema.safeParse(bad).success).toBe(false);
  });

  it('requires attribution', () => {
    const scene = buildScene('nob-hill-loop');
    expect(typeof scene.attribution).toBe('string');
    const bad = structuredClone(scene) as Record<string, unknown>;
    delete bad.attribution;
    expect(SceneDescriptionSchema.safeParse(bad).success).toBe(false);
  });
});

describe('Command', () => {
  it('accepts every documented command shape', () => {
    const commands = [
      { id: 'c1', cmd: 'set_paused', paused: true },
      { id: 'c2', cmd: 'step', frames: 3 },
      { id: 'c3', cmd: 'reset' },
      { id: 'c4', cmd: 'load_scenario', scenario_id: 'hyde-descent' },
      { id: 'c4b', cmd: 'load_location', query: 'Nob Hill', radius_m: 400 },
      { id: 'c5', cmd: 'set_param', key: 'cutin_period_s', value: 12 },
      { id: 'c6', cmd: 'toggle_layer', layer: 'detections', visible: false },
      { id: 'c7', cmd: 'set_camera', view: 'overhead' },
      { id: 'c8', cmd: 'inject_hazard', kind: 'cutin' },
    ];
    for (const c of commands) {
      const res = parseCommand(c);
      expect(res.ok, `${c.cmd}: ${res.ok ? '' : res.error}`).toBe(true);
    }
  });

  it('rejects an unknown layer key', () => {
    const res = parseCommand({
      id: 'x',
      cmd: 'toggle_layer',
      layer: 'nope',
      visible: true,
    });
    expect(res.ok).toBe(false);
  });

  it('requires a correlation id', () => {
    expect(CommandSchema.safeParse({ cmd: 'reset' }).success).toBe(false);
  });
});

describe('Ack', () => {
  it('parses and routes through the envelope', () => {
    const ack = {
      type: 'ack' as const,
      protocol: PROTOCOL_VERSION,
      id: 'c1',
      cmd: 'set_paused',
      ok: true,
      message: 'paused',
      t: 12.5,
    };
    expect(AckSchema.parse(ack)).toEqual(ack);
    const res = parseServerMessage(ack);
    expect(res.ok && res.value.type).toBe('ack');
  });
});
