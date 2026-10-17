// @vitest-environment jsdom
/**
 * Cockpit interior: the driver's-eye mount, the steering wheel and head motion
 * following the car, and the cabin staying out of every camera but the
 * cockpit view.
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { Cabin, EYE, blinkOn } from '../src/three/cabin';
import { ChaseCamera } from '../src/three/chaseCam';
import { EgoVehicle } from '../src/three/ego';
import { SCREEN_LAYER_MASK } from '../src/three/fsdScreen';
import { CH, DETECTOR_LAYER_MASK, applyMainLayers, assignChannel } from '../src/three/layers';
import { LAYER_KEYS } from '../src/schema';
import type { LayerKey, Pose } from '../src/schema';

const pose = (x: number, y: number, heading: number): Pose => ({ x, y, heading });
const allOn = Object.fromEntries(LAYER_KEYS.map((k: LayerKey) => [k, true])) as Record<LayerKey, boolean>;

/** Ego + cabin wired the way Renderer.tsx does it. */
function rig() {
  const ego = new EgoVehicle({ length: 4.9, width: 1.96, height: 1.44 });
  assignChannel(ego.group, CH.EGO);
  const cabin = new Cabin();
  ego.mesh.add(cabin.group);
  const cam = new ChaseCamera(16 / 9);
  cam.setCockpitMount(cabin.eyeAnchor);
  return { ego, cabin, cam };
}

const forward = (c: THREE.Camera) => new THREE.Vector3(0, 0, -1).applyQuaternion(c.quaternion);

describe('cockpit mount', () => {
  it("sits at the driver's eye, left of centre, looking ahead", () => {
    const { ego, cam } = rig();
    const p = pose(10, -4, 0);
    ego.setPose(p);
    cam.update(p, 8, 'cockpit', 1 / 60);
    const pos = cam.camera.position;
    // Sim (x, y) -> three (x, -y); the driver is at -Z (the car's left).
    expect(pos.x).toBeCloseTo(10 + EYE.x, 2);
    expect(pos.y).toBeCloseTo(EYE.y, 2);
    expect(pos.z).toBeCloseTo(4 + EYE.z, 2);
    const f = forward(cam.camera);
    expect(f.x).toBeGreaterThan(0.99);
    expect(f.y).toBeLessThan(0); // a touch below the horizon
    expect(cam.camera.near).toBe(0.15);
  });

  it('turns with the heading', () => {
    const { ego, cam } = rig();
    const p = pose(0, 0, Math.PI / 2);
    ego.setPose(p);
    cam.update(p, 0, 'cockpit', 1 / 60);
    expect(forward(cam.camera).z).toBeLessThan(-0.99); // north is three -z
  });

  it('pitches with the car on a hill', () => {
    const { ego, cam } = rig();
    const p = pose(0, 0, 0);
    ego.setPose(p);
    cam.update(p, 0, 'cockpit', 1 / 60);
    const flat = forward(cam.camera).y;
    ego.setPose(p, (x) => 0.15 * x); // climbing ahead
    cam.update(p, 0, 'cockpit', 1 / 60);
    expect(forward(cam.camera).y).toBeGreaterThan(flat + 0.1);
  });

  it('restores the coarse near plane outside the cockpit', () => {
    const { ego, cam } = rig();
    const p = pose(0, 0, 0);
    ego.setPose(p);
    cam.update(p, 0, 'cockpit', 1 / 60);
    cam.update(p, 0, 'chase', 1 / 60);
    expect(cam.camera.near).toBe(0.3);
  });
});

describe('cabin animation', () => {
  /** World position of the wheel's top-dead-centre marker. */
  function wheelTop(cabin: Cabin): THREE.Vector3 {
    cabin.group.updateWorldMatrix(true, true);
    return cabin.wheel.localToWorld(new THREE.Vector3(0, 0.18, 0));
  }

  it('turns the wheel left (toward -Z) for a left steer', () => {
    const { cabin } = rig();
    cabin.update(0, 0, 0, 'off', 0, 1 / 60);
    const straight = wheelTop(cabin);
    cabin.update(0.05, 0, 0, 'off', 0, 1 / 60);
    expect(wheelTop(cabin).z).toBeLessThan(straight.z - 0.05);
    cabin.update(-0.05, 0, 0, 'off', 0, 1 / 60);
    expect(wheelTop(cabin).z).toBeGreaterThan(straight.z + 0.05);
  });

  it('throws the head forward and down under braking, and back under throttle', () => {
    const { cabin } = rig();
    for (let i = 0; i < 120; i++) cabin.update(0, -6, 10, 'off', i / 60, 1 / 60);
    expect(cabin.eyeAnchor.position.x).toBeGreaterThan(EYE.x + 0.02);
    expect(cabin.eyeAnchor.rotation.x).toBeLessThan(EYE.pitch);
    for (let i = 0; i < 120; i++) cabin.update(0, 3, 10, 'off', i / 60, 1 / 60);
    expect(cabin.eyeAnchor.position.x).toBeLessThan(EYE.x);
  });

  it('leans the head outward in a corner', () => {
    const { cabin } = rig();
    // Left turn at speed: thrown to the right (+Z).
    for (let i = 0; i < 120; i++) cabin.update(0.1, 0, 12, 'off', i / 60, 1 / 60);
    expect(cabin.eyeAnchor.position.z).toBeGreaterThan(EYE.z + 0.01);
  });

  it('blinks at 1.5 Hz, half on', () => {
    expect(blinkOn(0)).toBe(true);
    expect(blinkOn(0.3)).toBe(true);
    expect(blinkOn(0.34)).toBe(false);
    expect(blinkOn(0.67)).toBe(true);
  });
});

describe('cabin layers', () => {
  it('is never in a detector frame', () => {
    expect(DETECTOR_LAYER_MASK & (1 << CH.CABIN)).toBe(0);
    const { cabin } = rig();
    const det = new THREE.PerspectiveCamera();
    det.layers.mask = DETECTOR_LAYER_MASK;
    cabin.group.traverse((o) => expect(o.layers.test(det.layers)).toBe(false));
  });

  it('keeps the centre screen from rendering itself, and shows the car on it', () => {
    expect(SCREEN_LAYER_MASK & (1 << CH.CABIN)).toBe(0);
    expect(SCREEN_LAYER_MASK & (1 << CH.EGO)).not.toBe(0);
    for (const c of [CH.BUILDINGS, CH.TREES, CH.LABELS]) expect(SCREEN_LAYER_MASK & (1 << c)).toBe(0);
  });

  it('swaps the exterior for the interior in cockpit view only', () => {
    const { ego, cabin } = rig();
    const main = new THREE.PerspectiveCamera();
    const shell = ego.mesh;

    applyMainLayers(main, allOn, 'cockpit');
    expect(shell.layers.test(main.layers)).toBe(false);
    cabin.group.traverse((o) => expect(o.layers.test(main.layers)).toBe(true));

    for (const view of ['chase', 'overhead', 'free'] as const) {
      applyMainLayers(main, allOn, view);
      expect(shell.layers.test(main.layers)).toBe(true);
      expect(cabin.screen.layers.test(main.layers)).toBe(false);
    }
  });
});
