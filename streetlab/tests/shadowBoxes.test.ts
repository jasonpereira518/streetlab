// @vitest-environment jsdom
/**
 * `createShadowBoxes` draws the perception source that is NOT driving the
 * car (`detections_shadow`) as unfilled wireframe outlines. It must treat
 * `null` (no second source running -- the default) and `[]` (the other
 * source ran and saw nothing) as distinct facts that both render nothing,
 * and it must pool its meshes rather than allocate one per detection per
 * frame (see hazardOverlay.ts / agents.ts for the established pattern).
 */
import { describe, expect, it } from 'vitest';
import * as THREE from 'three/webgpu';
import { createShadowBoxes } from '../src/three/shadowBoxes';
import type { Detection } from '../src/schema';

function drawables(root: THREE.Object3D): THREE.Object3D[] {
  const out: THREE.Object3D[] = [];
  root.traverse((o) => {
    if ((o as THREE.Mesh).isMesh) out.push(o);
  });
  return out;
}

function detection(id: string, x = 0, y = 0): Detection {
  return {
    id,
    cls: 'car',
    pose: { x, y, heading: 0 },
    size: { length: 4.6, width: 1.9, height: 1.46 },
    velocity: [0, 0],
    speed_mps: 0,
    confidence: 0.9,
    hazard: false,
    hazard_label: null,
    ttc_s: null,
    lane_offset: 0,
  };
}

/** Every shadow box ever allocated, visible or not (the pool's high-water mark). */
function poolSize(scene: THREE.Scene): number {
  const group = scene.getObjectByName('shadow-detections')!;
  return drawables(group).length;
}

function visibleCount(scene: THREE.Scene): number {
  const group = scene.getObjectByName('shadow-detections')!;
  return drawables(group).filter((o) => o.visible).length;
}

describe('createShadowBoxes', () => {
  it('draws nothing for null -- no second source running -- without throwing', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    expect(() => shadowBoxes.update(null)).not.toThrow();
    expect(visibleCount(scene)).toBe(0);

    shadowBoxes.dispose();
  });

  it('draws nothing for [] -- the other source ran and saw nothing -- distinct from null but also empty', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    expect(() => shadowBoxes.update([])).not.toThrow();
    expect(visibleCount(scene)).toBe(0);

    shadowBoxes.dispose();
  });

  it('adds exactly one visible object for one detection', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    shadowBoxes.update([detection('a', 3, -5)]);
    expect(visibleCount(scene)).toBe(1);

    const box = drawables(scene.getObjectByName('shadow-detections')!)[0] as THREE.Mesh;
    expect(box.position.x).toBeCloseTo(3, 5);
    expect(box.position.z).toBeCloseTo(5, 5);

    shadowBoxes.dispose();
  });

  it('does not leak objects as the list shrinks: pins the pool after 3 -> 1', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    shadowBoxes.update([detection('a'), detection('b'), detection('c')]);
    expect(poolSize(scene)).toBe(3);
    expect(visibleCount(scene)).toBe(3);

    shadowBoxes.update([detection('a')]);
    // The pool keeps its high-water mark -- hidden, not destroyed -- rather
    // than allocating fresh meshes the next time the list grows back.
    expect(poolSize(scene)).toBe(3);
    expect(visibleCount(scene)).toBe(1);

    // Cycling repeatedly must not grow the pool beyond the high-water mark.
    for (let i = 0; i < 20; i++) {
      shadowBoxes.update(i % 2 === 0 ? [detection('a'), detection('b'), detection('c')] : [detection('a')]);
    }
    expect(poolSize(scene)).toBe(3);

    shadowBoxes.dispose();
  });

  it('gates visibility through setVisible without disturbing pooled state', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    shadowBoxes.update([detection('a')]);
    shadowBoxes.setVisible(false);
    expect(scene.getObjectByName('shadow-detections')!.visible).toBe(false);

    shadowBoxes.setVisible(true);
    expect(scene.getObjectByName('shadow-detections')!.visible).toBe(true);
    expect(visibleCount(scene)).toBe(1);

    shadowBoxes.dispose();
  });

  it('dispose releases what it made', () => {
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);

    shadowBoxes.update([detection('a'), detection('b')]);
    const group = scene.getObjectByName('shadow-detections');
    expect(group).toBeTruthy();

    expect(() => shadowBoxes.dispose()).not.toThrow();
    expect(scene.getObjectByName('shadow-detections')).toBeUndefined();
  });
});
