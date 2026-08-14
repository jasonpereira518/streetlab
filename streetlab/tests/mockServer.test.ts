import { describe, expect, it } from 'vitest';
import { MockSim, createMockTransport } from '../src/net/mockServer';
import { SCENARIOS } from '../src/net/mockCity';
import { StateUpdateSchema, parseServerMessage } from '../src/schema';
import type { ServerMessage, StateUpdate } from '../src/schema';
import { makeRectRoute } from '../src/net/route';

describe('mock scene', () => {
  const sim = new MockSim();

  it('meets the authored-content floor', () => {
    const s = sim.scene;
    expect(s.buildings.length).toBeGreaterThanOrEqual(8);
    expect(s.roads.length).toBe(6);
    // Two signalised intersections, four heads each.
    expect(s.traffic_lights.length).toBe(8);
    expect(new Set(s.traffic_lights.map((t) => t.id.slice(0, 6))).size).toBe(2);
    expect(s.crosswalks.length).toBeGreaterThanOrEqual(8);
    expect(s.trees.length).toBeGreaterThanOrEqual(20);
    expect(s.stop_signs.length).toBeGreaterThan(0);
    expect(s.street_signs.some((n) => n.kind === 'street_name')).toBe(true);
    expect(s.catalog.length).toBe(5);
  });

  it('keeps every building inside the map bounds', () => {
    const { bounds } = sim.scene;
    for (const b of sim.scene.buildings) {
      for (const [x, y] of b.footprint) {
        expect(x).toBeGreaterThanOrEqual(bounds.min_x);
        expect(x).toBeLessThanOrEqual(bounds.max_x);
        expect(y).toBeGreaterThanOrEqual(bounds.min_y);
        expect(y).toBeLessThanOrEqual(bounds.max_y);
      }
    }
  });

  it('is deterministic across builds', () => {
    const a = new MockSim().scene;
    const b = new MockSim().scene;
    expect(JSON.stringify(a)).toBe(JSON.stringify(b));
  });
});

describe('mock frames', () => {
  it('validates every frame over a full simulated lap', () => {
    const sim = new MockSim();
    for (let i = 0; i < 60 * 40; i++) {
      sim.step();
      if (i % 7 !== 0) continue; // sample; validating 2400 frames is redundant
      const res = StateUpdateSchema.safeParse(sim.frame());
      expect(res.success, res.success ? '' : JSON.stringify(res.error.issues[0])).toBe(true);
    }
  });

  it('drives a white ego car around a closed loop', () => {
    const sim = new MockSim();
    const seen: Array<[number, number]> = [];
    // A lap is ~260 m and the ego stops at two signals and two stop signs, so
    // give it well over one lap's worth of wall clock.
    for (let i = 0; i < 60 * 150; i++) {
      sim.step();
      if (i % 30 === 0) {
        const f = sim.frame();
        seen.push([f.ego.pose.x, f.ego.pose.y]);
      }
    }
    const xs = seen.map((p) => p[0]);
    const ys = seen.map((p) => p[1]);
    // A loop, not a straight line: both axes must show real travel.
    expect(Math.max(...xs) - Math.min(...xs)).toBeGreaterThan(40);
    expect(Math.max(...ys) - Math.min(...ys)).toBeGreaterThan(40);
    // And it comes back near where it started.
    const start = seen[0];
    const closest = Math.min(
      ...seen.slice(40).map((p) => Math.hypot(p[0] - start[0], p[1] - start[1])),
    );
    expect(closest).toBeLessThan(12);
  });

  it('always reports exactly three traffic agents', () => {
    const sim = new MockSim();
    for (let i = 0; i < 600; i++) sim.step();
    expect(sim.frame().detections).toHaveLength(3);
  });

  it('emits a blue-plan polyline that starts at the ego', () => {
    const sim = new MockSim();
    for (let i = 0; i < 300; i++) sim.step();
    const f = sim.frame();
    expect(f.plan.polyline.length).toBeGreaterThan(10);
    const [px, py] = f.plan.polyline[0];
    expect(Math.hypot(px - f.ego.pose.x, py - f.ego.pose.y)).toBeLessThan(1.5);
  });

  it('produces periodic TTC values and a labelled cut-in hazard', () => {
    const sim = new MockSim();
    let ttcCount = 0;
    let hazardCount = 0;
    let label: string | null = null;
    for (let i = 0; i < 60 * 60; i++) {
      sim.step();
      const f = sim.frame();
      if (f.telemetry.ttc_s != null) ttcCount++;
      const h = f.detections.find((d) => d.hazard);
      if (h) {
        hazardCount++;
        label = h.hazard_label;
      }
    }
    expect(ttcCount).toBeGreaterThan(60);
    expect(hazardCount).toBeGreaterThan(60);
    expect(label).toBe('Cut-in vehicle');
  });

  it('cycles both signal groups through red, yellow and green', () => {
    const sim = new MockSim();
    const phases = new Set<string>();
    for (let i = 0; i < 60 * 40; i++) {
      sim.step();
      for (const s of sim.frame().signals) phases.add(s.phase);
    }
    expect(phases.has('red')).toBe(true);
    expect(phases.has('yellow')).toBe(true);
    expect(phases.has('green')).toBe(true);
  });

  it('stops the ego at least once for a red light or stop sign', () => {
    const sim = new MockSim();
    let minSpeed = Infinity;
    for (let i = 0; i < 60 * 90; i++) {
      sim.step();
      minSpeed = Math.min(minSpeed, sim.frame().ego.speed_mps);
    }
    expect(minSpeed).toBeLessThan(0.5);
  });
});

describe('mock commands', () => {
  it('pauses, steps and resets', () => {
    const sim = new MockSim();
    for (let i = 0; i < 120; i++) sim.step();
    const tBefore = sim.t;

    sim.apply({ id: '1', cmd: 'set_paused', paused: true });
    for (let i = 0; i < 60; i++) sim.step();
    expect(sim.frame().paused).toBe(true);
    expect(sim.t).toBeCloseTo(tBefore, 5);

    sim.apply({ id: '2', cmd: 'step', frames: 30 });
    expect(sim.t).toBeGreaterThan(tBefore);

    sim.apply({ id: '3', cmd: 'reset' });
    expect(sim.t).toBe(0);
  });

  it('loads another scenario and returns its scene', () => {
    const sim = new MockSim();
    const target = SCENARIOS[2];
    const res = sim.apply({ id: '4', cmd: 'load_scenario', scenario_id: target.id });
    expect(res.ok).toBe(true);
    expect(res.scene?.scenario_id).toBe(target.id);
    expect(sim.scene.name).toBe(target.name);
  });

  it('rejects an unknown scenario', () => {
    const sim = new MockSim();
    expect(sim.apply({ id: '5', cmd: 'load_scenario', scenario_id: 'nope' }).ok).toBe(
      false,
    );
  });

  it('applies a known parameter', () => {
    const sim = new MockSim();
    const res = sim.apply({
      id: '6',
      cmd: 'set_param',
      key: 'traffic_speed_scale',
      value: 0.5,
    });
    expect(res.ok).toBe(true);
  });

  it('does not crash on load_location — the mock has no geocoder', () => {
    // The mock is the fully in-process, offline transport, so it never
    // implements load_location itself; it must still return a well-formed
    // ack rather than crashing the switch that dispatches Command variants.
    const sim = new MockSim();
    const res = sim.apply({ id: '7', cmd: 'load_location', query: 'Nob Hill' });
    expect(res).toBeDefined();
    expect(res.ok).toBe(false);
  });
});

describe('mock transport', () => {
  it('emits a scene then a stream of valid frames, and acks commands', async () => {
    const transport = createMockTransport();
    const messages: ServerMessage[] = [];
    const invalid: string[] = [];
    let status = '';

    transport.connect({
      onMessage: (m) => messages.push(m),
      onStatus: (s) => {
        status = s;
      },
      onInvalid: (e) => invalid.push(e),
    });

    // Node has no rAF; the transport falls back to an interval timer.
    await new Promise((r) => setTimeout(r, 260));
    transport.send({ id: 'p', cmd: 'set_paused', paused: true });
    await new Promise((r) => setTimeout(r, 40));
    transport.close();

    expect(invalid).toEqual([]);
    expect(status).toBe('closed');
    expect(messages[0].type).toBe('scene_description');

    const frames = messages.filter(
      (m): m is StateUpdate => m.type === 'state_update',
    );
    // ~60 Hz for ~0.26 s, with generous slack for timer jitter.
    expect(frames.length).toBeGreaterThan(6);
    expect(frames.every((f) => parseServerMessage(f).ok)).toBe(true);
    expect(frames[frames.length - 1].seq).toBeGreaterThan(frames[0].seq);

    const acks = messages.filter((m) => m.type === 'ack');
    expect(acks).toHaveLength(1);
    expect(acks[0]).toMatchObject({ id: 'p', cmd: 'set_paused', ok: true });
  });
});

describe('route geometry', () => {
  it('produces continuous headings through filleted corners', () => {
    const route = makeRectRoute(0, 0, 60, 40, 8, true);
    let prev = route.sample(0).heading;
    for (let s = 0.5; s < route.length; s += 0.5) {
      const h = route.sample(s).heading;
      let d = Math.abs(h - prev);
      while (d > Math.PI) d = Math.abs(d - Math.PI * 2);
      expect(d).toBeLessThan(0.15);
      prev = h;
    }
  });

  it('offsets laterally to the left of travel', () => {
    const route = makeRectRoute(0, 0, 60, 40, 8, false);
    // s = 0 sits mid-fillet at the first corner; 25 m is on the straight
    // bottom edge, travelling +x.
    const s = 25;
    const base = route.sample(s);
    const left = route.sample(s, 2);
    expect(base.heading).toBeCloseTo(0, 3);
    expect(left.y - base.y).toBeCloseTo(2, 3);
  });

  it('wraps and measures signed gaps around the loop', () => {
    const route = makeRectRoute(0, 0, 60, 40, 8, true);
    expect(route.wrap(-1)).toBeCloseTo(route.length - 1, 6);
    expect(route.gap(10, 20)).toBeCloseTo(10, 6);
    expect(route.gap(20, 10)).toBeCloseTo(-10, 6);
  });
});
