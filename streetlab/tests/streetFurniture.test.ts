// @vitest-environment jsdom
/**
 * Where street furniture geometry actually lands.
 *
 * These read the built scene graph rather than re-deriving the placement, so
 * they fail when the mesh a user looks at is wrong, not merely when a formula
 * changed.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { buildWorld } from '../src/three/world';
import type { SceneDescription } from '../src/schema';

const gridScene = (): SceneDescription =>
  JSON.parse(
    readFileSync(resolve(__dirname, '../../contract/fixtures/scene_description.json'), 'utf8'),
  );

/** `(x, z)` in three-space of instance `i`. */
function instanceXZ(mesh: THREE.InstancedMesh, i: number): [number, number] {
  const m = new THREE.Matrix4();
  mesh.getMatrixAt(i, m);
  return [m.elements[12], m.elements[14]];
}

/** Half-width of a geometry on the ground plane, i.e. its widest radius. */
function groundRadius(geo: THREE.BufferGeometry): number {
  geo.computeBoundingBox();
  const b = geo.boundingBox!;
  return Math.max(Math.abs(b.min.x), Math.abs(b.max.x), Math.abs(b.min.z), Math.abs(b.max.z));
}

describe('stop sign posts', () => {
  const world = buildWorld(gridScene());
  const group = world.root.getObjectByName('stop-signs') as THREE.Group;
  const [posts, octs, faces] = group.children as THREE.InstancedMesh[];

  it('has a post, a plate and a face per sign', () => {
    expect(posts.count).toBeGreaterThan(0);
    expect(octs.count).toBe(posts.count);
    expect(faces.count).toBe(posts.count);
  });

  it('keeps the post entirely behind the sign face', () => {
    /**
     * The post used to run straight up through the middle of the octagon at
     * the same x/z as the plate. Its radius (0.05 m) is larger than the gap
     * the face was floated in front of the plate by (0.028 m), so a strip of
     * grey post was drawn over the middle of every red octagon.
     */
    const postR = groundRadius(posts.geometry);
    for (let i = 0; i < posts.count; i++) {
      const heading = gridScene().stop_signs[i].heading;
      // Three-space forward, for a world heading: world (cos h, sin h) -> (x, -z).
      const fx = Math.cos(heading);
      const fz = -Math.sin(heading);
      const [px, pz] = instanceXZ(posts, i);
      const [ox, oz] = instanceXZ(octs, i);
      const [ax, az] = instanceXZ(faces, i);
      const postAhead = (px - ox) * fx + (pz - oz) * fz;
      const faceAhead = (ax - ox) * fx + (az - oz) * fz;
      expect(postAhead + postR).toBeLessThan(faceAhead);
    }
  });
});


describe('street sign posts', () => {
  const scene = gridScene();
  const world = buildWorld(scene);
  const group = world.root.getObjectByName('street-signs') as THREE.Group;
  const postMesh = group.children[0] as THREE.InstancedMesh;
  const blades = group.children.slice(1) as THREE.Mesh[];

  it('mounts a blade for every street sign', () => {
    expect(blades.length).toBe(scene.street_signs.length);
    expect(postMesh.count).toBeGreaterThan(0);
  });

  it('keeps the post out of the face of every blade it carries', () => {
    /**
     * Same defect as the stop sign, smaller: a blade centred on its own post
     * is bisected by it, and because the blade is a zero-thickness plane the
     * post's front half draws over the text.
     */
    const postR = groundRadius(postMesh.geometry);
    const postXZ = Array.from({ length: postMesh.count }, (_, i) => instanceXZ(postMesh, i));
    for (const blade of blades) {
      // The blade's plane faces along its own +x, rotated by `rotation.y`.
      const fx = Math.cos(blade.rotation.y);
      const fz = -Math.sin(blade.rotation.y);
      const near = postXZ
        .map(([x, z]) => (x - blade.position.x) * fx + (z - blade.position.z) * fz)
        .filter((_, i) => Math.hypot(postXZ[i][0] - blade.position.x, postXZ[i][1] - blade.position.z) < 1.0);
      expect(near.length).toBeGreaterThan(0);
      for (const ahead of near) {
        expect(ahead + postR).toBeLessThanOrEqual(0);
      }
    }
  });
});
