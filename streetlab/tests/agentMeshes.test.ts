// @vitest-environment jsdom
/**
 * The agent meshes are what the detector is scored on: people must be shaped like
 * people (not scaled vehicles), vehicles must keep the footprint labels are sized from,
 * and paint must be a pure function of the agent id.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { TrafficFleet } from '../src/three/agents';
import { buildCyclist, buildPedestrian } from '../src/three/figures';
import { agentMaterial, hashId } from '../src/three/meshKit';
import { buildVehicleGeometry, styleFor } from '../src/three/vehicles';

const bbox = (o: THREE.Object3D) => new THREE.Box3().setFromObject(o);

describe('vehicles', () => {
  it('keep the exact footprint and height the simulator reported', () => {
    for (const [cls, size] of [
      ['car', { length: 4.6, width: 1.9, height: 1.45 }],
      ['truck', { length: 7.8, width: 2.4, height: 3.1 }],
      ['bus', { length: 11.5, width: 2.55, height: 3.3 }],
    ] as const) {
      const g = buildVehicleGeometry(size, styleFor(cls, 'veh_00'));
      g.computeBoundingBox();
      const b = g.boundingBox!;
      // Lamps and plates stand a few cm proud of the body; labels are sized from `size`.
      expect(b.max.x - b.min.x).toBeGreaterThan(size.length - 0.05);
      expect(b.max.x - b.min.x).toBeLessThan(size.length + 0.25);
      expect(b.max.z - b.min.z).toBeLessThan(size.width + 0.2);
      expect(b.max.y).toBeLessThan(size.height + 0.15);
      expect(b.min.y).toBeGreaterThanOrEqual(-0.01);
    }
  });

  it('paint is a pure function of (class, id)', () => {
    expect(styleFor('car', 'veh_07')).toEqual(styleFor('car', 'veh_07'));
    const bodies = new Set(Array.from({ length: 40 }, (_, i) => styleFor('car', `veh_${i}`).body));
    expect(bodies.size).toBeGreaterThan(4);
    expect(hashId('a')).not.toBe(hashId('b'));
  });
});

describe('people', () => {
  it('a pedestrian is a 1.75 m humanoid with moving limbs, not a vehicle box', () => {
    const p = buildPedestrian('ped_1', agentMaterial());
    const b = bbox(p.group);
    expect(b.max.y - b.min.y).toBeGreaterThan(1.6);
    expect(b.max.y - b.min.y).toBeLessThan(1.85);
    expect(b.max.z - b.min.z).toBeLessThan(0.7);
    const before = p.group.children.slice(1).map((c) => c.rotation.z);
    p.animate(1.0, 1.4);
    const after = p.group.children.slice(1).map((c) => c.rotation.z);
    expect(after).not.toEqual(before);
    p.animate(1.0, 0);
    expect(p.group.children.slice(1).every((c) => c.rotation.z === 0)).toBe(true);
    p.dispose();
  });

  it('a cyclist is a bicycle plus rider about the reference size, and pedals', () => {
    const c = buildCyclist('cyc_1', agentMaterial());
    c.animate(0, 4);
    const b = bbox(c.group);
    expect(b.max.x - b.min.x).toBeGreaterThan(1.4);
    expect(b.max.x - b.min.x).toBeLessThan(2.0);
    expect(b.max.y).toBeGreaterThan(1.5);
    expect(b.max.y).toBeLessThan(1.8);
    const legs = c.group.children.slice(1);
    const pos0 = legs.map((l) => l.position.y);
    c.animate(1.3, 4);
    expect(legs.map((l) => l.position.y)).not.toEqual(pos0);
    c.dispose();
  });
});

describe('TrafficFleet', () => {
  it('draws pedestrians and cyclists as figures, at the size the frame reports', () => {
    const fleet = new TrafficFleet();
    const pose = { x: 5, y: 0, heading: 0 };
    fleet.update(
      [
        { id: 'p', cls: 'pedestrian', pose, size: { length: 0.6, width: 0.6, height: 1.75 }, speed_mps: 1.4 },
        { id: 'c', cls: 'cyclist', pose: { ...pose, y: 3 }, size: { length: 1.8, width: 0.7, height: 1.7 }, speed_mps: 4 },
        { id: 'm', cls: 'motorcycle', pose: { ...pose, y: 6 }, size: { length: 2.1, width: 0.8, height: 1.3 }, speed_mps: 8 },
      ],
      0.016,
    );
    fleet.group.updateMatrixWorld(true);
    const [ped, cyc, moto] = fleet.group.children;
    expect(bbox(ped).max.y).toBeGreaterThan(1.6);
    expect(bbox(ped).max.y).toBeLessThan(1.85);
    expect(bbox(cyc).max.y).toBeLessThan(1.85);
    expect(bbox(moto).max.y).toBeGreaterThan(1.2);
    fleet.dispose();
  });
});
