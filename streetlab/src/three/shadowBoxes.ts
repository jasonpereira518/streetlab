/**
 * Wireframe outlines for `detections_shadow` -- the perception source that is
 * NOT driving the car this frame (see `schema.ts`). Two sources run in
 * shadow mode; the driving one is drawn as solid vehicles by `agents.ts`, and
 * this module draws the other as unfilled outlines so the two readings stay
 * visually distinct at a glance. A shadow box with no vehicle inside it is a
 * false positive; a vehicle with no shadow box is a miss.
 *
 * `detections_shadow` is nullable-not-optional and the two empty-ish values
 * mean different things: `null` is "no second source running" (the default
 * configuration, so this must be the safe, no-throw path), `[]` is "the
 * other source ran and saw nothing." Both draw nothing.
 *
 * Pooling mirrors `hazardOverlay.ts`: one shared box geometry/material, and
 * an index-addressed slot array that grows on demand and hides (never
 * destroys) entries when the incoming list shrinks, so per-frame churn in
 * the detection count never allocates or leaks GPU resources.
 */
import * as THREE from 'three/webgpu';
import { float, min, smoothstep, uniform, uv } from 'three/tsl';
import type { Detection } from '../schema';

/** Distinct from the hazard overlay's orange and the plan ribbon's blue. */
const OUTLINE_COLOR = '#8B5CF6';
/** Boxes are padded slightly so the outline reads as an annotation, not coplanar bodywork. */
const PAD = 0.08;

export function createShadowBoxes(scene: THREE.Scene) {
  const group = new THREE.Group();
  group.name = 'shadow-detections';
  scene.add(group);

  const boxGeo = new THREE.BoxGeometry(1, 1, 1);
  const boxMat = new THREE.MeshBasicNodeMaterial({
    transparent: true,
    depthWrite: false,
    side: THREE.DoubleSide,
    toneMapped: false,
  });

  // Distance to the nearest edge of each face, in UV space -- same
  // edge-highlight trick as hazardOverlay.ts, tightened and with no interior
  // fill, so the box reads as an unfilled wireframe outline rather than a
  // translucent solid.
  const d = min(min(uv().x, uv().x.oneMinus()), min(uv().y, uv().y.oneMinus()));
  const edge = smoothstep(float(0.035), float(0.008), d);
  boxMat.colorNode = uniform(new THREE.Color(OUTLINE_COLOR));
  boxMat.opacityNode = edge;

  const boxes: THREE.Mesh[] = [];

  function boxAt(i: number): THREE.Mesh {
    let box = boxes[i];
    if (box) return box;
    box = new THREE.Mesh(boxGeo, boxMat);
    box.renderOrder = 6;
    box.frustumCulled = false;
    group.add(box);
    boxes[i] = box;
    return box;
  }

  function update(detections: Detection[] | null): void {
    const list = detections ?? [];

    list.forEach((det, i) => {
      const box = boxAt(i);
      const x = det.pose.x;
      const z = -det.pose.y;

      box.visible = true;
      box.position.set(x, det.size.height / 2, z);
      box.rotation.y = det.pose.heading;
      box.scale.set(
        det.size.length + PAD * 2,
        det.size.height + PAD,
        det.size.width + PAD * 2,
      );
    });

    for (let i = list.length; i < boxes.length; i++) boxes[i].visible = false;
  }

  function setVisible(visible: boolean): void {
    group.visible = visible;
  }

  function dispose(): void {
    boxGeo.dispose();
    boxMat.dispose();
    boxes.length = 0;
    group.clear();
    scene.remove(group);
  }

  return { update, setVisible, dispose };
}
