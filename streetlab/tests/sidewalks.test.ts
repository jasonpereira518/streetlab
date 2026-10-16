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
import { roadHalfWidth } from '../src/schema';

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
    const half = roadHalfWidth(r);
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

describe('pavement is drawn only on the sides that have one', () => {
  /**
   * 16 of the extract's 264 ways say `sidewalk=right`. While the wire carried
   * a single `has_sidewalk` boolean the renderer had no way to know which side
   * that was, and laid a pavement down the one OSM says is not there.
   */
  function sidesOf(left: boolean, right: boolean): { left: number; right: number } {
    const scene: SceneDescription = {
      ...osmScene(),
      buildings: [],
      trees: [],
      crosswalks: [],
      stop_signs: [],
      traffic_lights: [],
      street_signs: [],
      kerb_returns: [],
      reference_path: [],
      roads: [
        {
          id: 'r',
          name: 'One Sided St',
          road_class: 'residential',
          centerline: [
            [0, -60],
            [0, 60],
          ],
          lanes_forward: 1,
          lanes_backward: 1,
          lane_width_m: 3.6,
          speed_limit_mps: 11.2,
          oneway: false,
          center_marking: 'broken_yellow',
          sidewalk_left: left,
          sidewalk_right: right,
          layer: 0,
          bridge: false,
          tunnel: false,
          edge_m: 0,
        },
      ],
    };
    const world = buildWorld(scene);
    const walk = world.root.getObjectByName('sidewalks') as THREE.Mesh;
    const pos = walk.geometry.getAttribute('position');
    let west = 0;
    let east = 0;
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      if (x < -0.5) west++;
      else if (x > 0.5) east++;
    }
    // Travel is north, so the driver's left is WEST -- `offsetAt` puts
    // +lateral at (-ty, +tx), which for a northbound tangent is -x.
    return { left: west, right: east };
  }

  it('draws both when both sides have one', () => {
    const both = sidesOf(true, true);
    expect(both.left).toBeGreaterThan(0);
    expect(both.right).toBeGreaterThan(0);
  });

  it('draws only the right when only the right has one', () => {
    const only = sidesOf(false, true);
    expect(only.right).toBeGreaterThan(0);
    expect(only.left).toBe(0);
  });

  it('draws only the left when only the left has one', () => {
    const only = sidesOf(true, false);
    expect(only.left).toBeGreaterThan(0);
    expect(only.right).toBe(0);
  });
});

/** How far inside a kerb return a point is (beyond its arc), or -Infinity. */
function depthIntoReturns(scene: SceneDescription, x: number, y: number): number {
  let worst = -Infinity;
  for (const kr of scene.kerb_returns) {
    const quad = [kr.corner, kr.tangent_a, kr.center, kr.tangent_b];
    let inside = true;
    let sign = 0;
    for (let i = 0; i < 4 && inside; i++) {
      const [ax, ay] = quad[i];
      const [bx, by] = quad[(i + 1) % 4];
      const cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax);
      if (Math.abs(cross) < 1e-9) continue;
      if (sign === 0) sign = Math.sign(cross);
      else if (Math.sign(cross) !== sign) inside = false;
    }
    if (!inside) continue;
    worst = Math.max(worst, Math.hypot(x - kr.center[0], y - kr.center[1]) - kr.radius_m);
  }
  return worst;
}

/** Whether any triangle of the named mesh covers the world point. */
function meshCovers(scene: SceneDescription, name: string, x: number, y: number): boolean {
  const mesh = buildWorld(scene).root.getObjectByName(name) as THREE.Mesh;
  const pos = mesh.geometry.getAttribute('position');
  const index = mesh.geometry.getIndex();
  const n = index ? index.count : pos.count;
  const at = (k: number) => {
    const i = index ? index.getX(k) : k;
    return [pos.getX(i), -pos.getZ(i)];
  };
  for (let k = 0; k < n; k += 3) {
    const [a, b, c] = [at(k), at(k + 1), at(k + 2)];
    const d1 = (x - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (y - b[1]);
    const d2 = (x - c[0]) * (b[1] - c[1]) - (b[0] - c[0]) * (y - c[1]);
    const d3 = (x - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (y - a[1]);
    const neg = d1 < 0 || d2 < 0 || d3 < 0;
    const pos_ = d1 > 0 || d2 > 0 || d3 > 0;
    if (!(neg && pos_)) return true;
  }
  return false;
}

describe('kerb returns', () => {
  /**
   * Junction corners used to be square, and a car turning on its 6 m lane
   * radius crossed the pavement corner by 0.46-0.52 m on every grid right
   * turn. The backend now rounds each corner (`KerbReturn`, 4.5 m) and both
   * placement and the renderer use those exact shapes.
   */
  it.each([
    ['the real Nob Hill extract', osmScene],
    ['the synthetic grid', gridScene],
  ])('pavement stays off them on %s', (_label, load) => {
    const scene = load();
    expect(scene.kerb_returns.length).toBeGreaterThan(0);
    const offenders = pavementSamples(scene)
      .map(([x, y]) => depthIntoReturns(scene, x, y))
      // Within 5 cm of the arc is the kerb line, drawn from both sides.
      .filter((d) => d > 0.05);
    expect(offenders.length).toBe(0);
  });

  it('each one on the grid is drawn as tarmac', () => {
    const scene = gridScene();
    const missing = scene.kerb_returns.filter((kr) => {
      // A point just off the old square corner: pavement before, road now.
      const x = kr.corner[0] + 0.15 * (kr.center[0] - kr.corner[0]);
      const y = kr.corner[1] + 0.15 * (kr.center[1] - kr.corner[1]);
      return !meshCovers(scene, 'roads', x, y);
    });
    expect(missing.length).toBe(0);
  });
});
