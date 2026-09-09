// @vitest-environment jsdom
/**
 * Pavement geometry, checked against the mesh the renderer actually builds.
 *
 * The fixture is the real Nob Hill extract trimmed to a 200 m box around the
 * origin (46 roads, 449 buildings), regenerated from
 * `streetlab-backend/tests/fixtures/overpass_nob_hill.json`. Its junctions are
 * skewed and irregular, which is the whole point: the synthetic grid's streets
 * cross at right angles and hide every bug that depends on the crossing angle.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { buildWorld } from '../src/three/world';
import type { SceneDescription } from '../src/schema';

const osmScene = (): SceneDescription =>
  JSON.parse(readFileSync(resolve(__dirname, 'fixtures/nobHillScene.json'), 'utf8'));
const gridScene = (): SceneDescription =>
  JSON.parse(
    readFileSync(resolve(__dirname, '../../contract/fixtures/scene_description.json'), 'utf8'),
  );

function segDist(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const dx = bx - ax;
  const dy = by - ay;
  const L = dx * dx + dy * dy;
  if (L < 1e-12) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / L));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

/** How far a world point lies inside the deepest carriageway covering it. */
function depthIntoRoad(scene: SceneDescription, x: number, y: number): number {
  let worst = -Infinity;
  for (const r of scene.roads) {
    const half = ((r.lanes_forward + r.lanes_backward) * r.lane_width_m) / 2;
    for (let i = 1; i < r.centerline.length; i++) {
      const a = r.centerline[i - 1];
      const b = r.centerline[i];
      worst = Math.max(worst, half - segDist(x, y, a[0], a[1], b[0], b[1]));
    }
  }
  return worst;
}

/** Every vertex and triangle centroid of the pavement mesh, in world metres. */
function pavementSamples(scene: SceneDescription): [number, number][] {
  const world = buildWorld(scene);
  const walk = world.root.getObjectByName('sidewalks') as THREE.Mesh;
  const pos = walk.geometry.getAttribute('position');
  const out: [number, number][] = [];
  for (let i = 0; i < pos.count; i += 3) {
    let cx = 0;
    let cy = 0;
    for (let k = 0; k < 3; k++) {
      const x = pos.getX(i + k);
      const y = -pos.getZ(i + k);
      out.push([x, y]);
      cx += x;
      cy += y;
    }
    out.push([cx / 3, cy / 3]);
  }
  return out;
}

describe('pavement never covers the carriageway', () => {
  /**
   * Measured before this: 19,436 of 101,552 sample points on the real extract
   * sat on tarmac, the worst 5.37 m inside a road. Two causes, both invisible
   * on a right-angled grid — the gap cut in a pavement where another street
   * crosses was sized by that street's half-width measured ALONG the arc,
   * which understates it as the crossing angle leaves 90 degrees; and the
   * corner infill was an axis-aligned box, which is simply the wrong shape
   * for a junction that is not square to the world axes.
   */
  it.each([
    ['the real Nob Hill extract', osmScene],
    ['the synthetic grid', gridScene],
  ])('%s', (_label, load) => {
    const scene = load();
    const offenders = pavementSamples(scene)
      .map(([x, y]) => depthIntoRoad(scene, x, y))
      // The kerb line itself is shared geometry: pavement is drawn from the
      // kerb outward, so touching it to within a millimetre is correct.
      .filter((d) => d > 1e-3);
    const worst = offenders.length ? Math.max(...offenders) : 0;
    expect({ count: offenders.length, worst: +worst.toFixed(2) }).toEqual({
      count: 0,
      worst: 0,
    });
  });
});
