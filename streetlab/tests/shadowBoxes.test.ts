// @vitest-environment jsdom
/**
 * `createShadowBoxes` draws the perception source that is NOT driving the
 * car (`detections_shadow`) as unfilled wireframe outlines. It must treat
 * `null` (no second source running -- the default) and `[]` (the other
 * source ran and saw nothing) as distinct facts that both render nothing,
 * and it must pool its meshes rather than allocate one per detection per
 * frame (see hazardOverlay.ts / agents.ts for the established pattern).
 */
import { describe, expect, it, vi } from 'vitest';
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

  it('draws an unfilled outline, not hazardOverlay-style filled/pulsing boxes', () => {
    // Pins the visual intent this task exists for: the shadow source must
    // read as a bare wireframe (opacity purely a function of distance-to-edge
    // via `smoothstep`), never a translucent fill with a base-opacity term or
    // a breathing pulse layered on top the way hazardOverlay.ts's hazard box
    // is. A `edge.mul(0.92).add(float(0.1))`-style regression (a plausible
    // copy-paste from hazardOverlay.ts) changes the node graph's shape, which
    // this test inspects directly rather than trusting a comment.
    const scene = new THREE.Scene();
    const shadowBoxes = createShadowBoxes(scene);
    shadowBoxes.update([detection('a')]);

    const box = drawables(scene.getObjectByName('shadow-detections')!)[0] as THREE.Mesh;
    const mat = box.material as THREE.MeshBasicNodeMaterial;

    expect(mat.transparent).toBe(true);

    // `opacityNode` must be exactly the smoothstep edge term -- no further
    // `.add()`/`.mul()` composition. Every TSL expression node is wrapped in
    // a `VarNode`; unwrap one level and the underlying node must still be the
    // `MathNode` smoothstep produces. `.add()`/`.mul()` on top of it (a fill
    // term or a pulse) would instead surface as an `OperatorNode` here.
    const inner = (mat.opacityNode as unknown as { node?: unknown }).node as
      | { constructor: { name: string }; method?: string }
      | undefined;
    expect(inner?.constructor.name).toBe('MathNode');
    expect(inner?.method).toBe('smoothstep');

    shadowBoxes.dispose();
  });

  it('dispose releases the GPU geometry and material it created', () => {
    // The brief's named failure mode is leaking GPU resources over a
    // session; asserting only that the group leaves the scene (as a prior
    // version of this test did) would pass even if `boxGeo.dispose()` and
    // `boxMat.dispose()` were deleted. Spy on the actual GPU-release calls.
    const scene = new THREE.Scene();
    const geoDisposeSpy = vi.spyOn(THREE.BoxGeometry.prototype, 'dispose');
    const matDisposeSpy = vi.spyOn(THREE.MeshBasicNodeMaterial.prototype, 'dispose');

    const shadowBoxes = createShadowBoxes(scene);
    shadowBoxes.update([detection('a'), detection('b')]);
    const group = scene.getObjectByName('shadow-detections');
    expect(group).toBeTruthy();

    expect(() => shadowBoxes.dispose()).not.toThrow();

    expect(scene.getObjectByName('shadow-detections')).toBeUndefined();
    expect(geoDisposeSpy).toHaveBeenCalledTimes(1);
    expect(matDisposeSpy).toHaveBeenCalledTimes(1);

    geoDisposeSpy.mockRestore();
    matDisposeSpy.mockRestore();
  });
});
