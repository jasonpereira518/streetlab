// @vitest-environment jsdom
/**
 * Location-independent render rules, checked against the meshes the renderer
 * builds.
 *
 * The real-extract fixtures are nearly all two-point roads and fewer than 50
 * of them, which hides two whole classes of bug: anything that only shows on
 * a road with bends, and anything that grows with a road's index in the
 * scene. So these scenes are built to have both.
 */
import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import * as THREE from 'three/webgpu';
import { buildWorld } from '../src/three/world';
import { TerrainField, attitudeOn } from '../src/three/terrain';
import type { Road, SceneDescription, Vec2 } from '../src/schema';

const gridScene = (): SceneDescription =>
  JSON.parse(
    readFileSync(resolve(__dirname, '../../contract/fixtures/scene_description.json'), 'utf8'),
  );

const half = (r: Road) => ((r.lanes_forward + r.lanes_backward) * r.lane_width_m) / 2;

function road(id: string, centerline: Vec2[], extra: Partial<Road> = {}): Road {
  return {
    id,
    name: id,
    road_class: 'residential',
    centerline,
    lanes_forward: 1,
    lanes_backward: 1,
    lane_width_m: 3.6,
    speed_limit_mps: 11,
    oneway: false,
    center_marking: 'broken_yellow',
    sidewalk_left: true,
    sidewalk_right: true,
    ...extra,
  };
}

/** A quarter-circle of radius 60 m sampled every 15 degrees. */
function arc(cx: number, cy: number, r: number): Vec2[] {
  const out: Vec2[] = [];
  for (let d = 0; d <= 90; d += 15) {
    const a = (d * Math.PI) / 180;
    out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]);
  }
  return out;
}

/**
 * A curved two-lane street and a curved arterial with an edge line, placed
 * LAST behind 300 short, far-apart filler streets so any height that climbs
 * with the road index has had room to climb.
 */
function curvyScene(): SceneDescription {
  const base = gridScene();
  const roads: Road[] = [];
  for (let i = 0; i < 300; i++) {
    const x = 2000 + (i % 20) * 40;
    const y = 2000 + Math.floor(i / 20) * 40;
    roads.push(road(`filler_${i}`, [[x, y], [x + 10, y]], { sidewalk_left: false, sidewalk_right: false }));
  }
  roads.push(road('curve_local', arc(0, 0, 60)));
  roads.push(
    road('curve_arterial', arc(-400, -400, 80), {
      road_class: 'arterial',
      lanes_forward: 2,
      lanes_backward: 2,
    }),
  );
  return {
    ...base,
    roads,
    buildings: [],
    crosswalks: [],
    traffic_lights: [],
    stop_signs: [],
    trees: [],
    street_signs: [],
  };
}

interface Tri {
  p: [number, number][];
  h: number[];
}

function tris(scene: SceneDescription, name: string): Tri[] {
  const world = buildWorld(scene);
  const mesh = world.root.getObjectByName(name) as THREE.Mesh;
  const pos = mesh.geometry.getAttribute('position');
  const out: Tri[] = [];
  for (let i = 0; i < pos.count; i += 3) {
    const p: [number, number][] = [];
    const h: number[] = [];
    for (let k = 0; k < 3; k++) {
      p.push([pos.getX(i + k), -pos.getZ(i + k)]);
      h.push(pos.getY(i + k));
    }
    out.push({ p, h });
  }
  return out;
}

function segDist(px: number, py: number, a: Vec2, b: Vec2) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const L = dx * dx + dy * dy;
  const t = L < 1e-12 ? 0 : Math.max(0, Math.min(1, ((px - a[0]) * dx + (py - a[1]) * dy) / L));
  return Math.hypot(px - (a[0] + t * dx), py - (a[1] + t * dy));
}

function distToLine(line: Vec2[], x: number, y: number) {
  let best = Infinity;
  for (let i = 1; i < line.length; i++) best = Math.min(best, segDist(x, y, line[i - 1], line[i]));
  return best;
}

function inTri(x: number, y: number, [a, b, c]: [number, number][]) {
  const s = (p: [number, number], q: [number, number]) =>
    (q[0] - p[0]) * (y - p[1]) - (q[1] - p[1]) * (x - p[0]);
  const d1 = s(a, b);
  const d2 = s(b, c);
  const d3 = s(c, a);
  const neg = d1 < -1e-9 || d2 < -1e-9 || d3 < -1e-9;
  const pos = d1 > 1e-9 || d2 > 1e-9 || d3 > 1e-9;
  return !(neg && pos);
}

/** Points along a polyline every `step` metres, at a lateral offset. */
function samples(line: Vec2[], lateral: number, step = 1): Vec2[] {
  const out: Vec2[] = [];
  for (let i = 1; i < line.length; i++) {
    const [ax, ay] = line[i - 1];
    const [bx, by] = line[i];
    const len = Math.hypot(bx - ax, by - ay);
    const tx = (bx - ax) / len;
    const ty = (by - ay) / len;
    // Mid-step, so no sample sits exactly on a quad's end edge.
    for (let s = step / 2; s < len; s += step) {
      out.push([ax + tx * s - ty * lateral, ay + ty * s + tx * lateral]);
    }
  }
  return out;
}

const curved = (scene: SceneDescription) => scene.roads.filter((r) => r.id.startsWith('curve_'));

describe('the road surface follows its centreline', () => {
  it('covers the whole carriageway of a curved road, not the chord between its ends', () => {
    const scene = curvyScene();
    const surface = tris(scene, 'roads');
    for (const r of curved(scene)) {
      // Just inside each kerb, and on the centre.
      for (const lat of [0, half(r) - 0.1, -(half(r) - 0.1)]) {
        const missing = samples(r.centerline, lat).filter(
          ([x, y]) => !surface.some((t) => inTri(x, y, t.p)),
        );
        expect(missing, `${r.id} lateral ${lat}`).toHaveLength(0);
      }
    }
  });

  it('never paints tarmac outside the carriageway', () => {
    const scene = curvyScene();
    for (const t of tris(scene, 'roads')) {
      for (const [x, y] of t.p) {
        // A mitred corner overshoots a round one by half*(sec(turn/2) - 1),
        // for these 15-degree bends under 1% of the half-width.
        const miter = 1 / Math.cos((7.5 * Math.PI) / 180);
        const worst = Math.min(
          ...scene.roads.map((r) => distToLine(r.centerline, x, y) - half(r) * miter),
        );
        expect(worst).toBeLessThan(0.01);
      }
    }
  });
});

describe('lane markings line up with their road', () => {
  it('keeps every painted line inside a carriageway, curves included', () => {
    const scene = curvyScene();
    const outside = tris(scene, 'lane-markings').filter((t) => {
      const cx = (t.p[0][0] + t.p[1][0] + t.p[2][0]) / 3;
      const cy = (t.p[0][1] + t.p[1][1] + t.p[2][1]) / 3;
      return Math.min(...scene.roads.map((r) => distToLine(r.centerline, cx, cy) - half(r))) > 0;
    });
    expect(outside).toHaveLength(0);
  });

  it('runs the centre line along the centre of a curved road', () => {
    const scene = curvyScene();
    const r = scene.roads.find((x) => x.id === 'curve_local')!;
    const yellow = tris(scene, 'lane-markings').filter((t) => {
      const [x, y] = t.p[0];
      return Math.hypot(x, y) < 100; // the local arc is centred on the origin
    });
    expect(yellow.length).toBeGreaterThan(0);
    for (const t of yellow) {
      for (const [x, y] of t.p) {
        expect(distToLine(r.centerline, x, y)).toBeLessThan(0.1);
      }
    }
  });
});

describe('the height stack does not depend on how many roads there are', () => {
  it('keeps tarmac below paint, crossings and pavement', () => {
    const scene = curvyScene();
    const top = (name: string) => Math.max(...tris(scene, name).flatMap((t) => t.h));
    const bottom = (name: string) => Math.min(...tris(scene, name).flatMap((t) => t.h));
    const roadTop = top('roads');
    expect(roadTop).toBeLessThan(bottom('lane-markings'));
    // The pavement's walking surface, not its kerb face, which starts at road level.
    expect(roadTop).toBeLessThan(top('sidewalks'));
    expect(roadTop).toBeLessThan(0.03);
  });

  it('keeps the grid crossings above the tarmac too', () => {
    const scene = gridScene();
    const roadTop = Math.max(...tris(scene, 'roads').flatMap((t) => t.h));
    const walkBottom = Math.min(...tris(scene, 'crosswalks').flatMap((t) => t.h));
    expect(roadTop).toBeLessThan(walkBottom);
  });
});

describe('junctions', () => {
  /** Two ways of one street meeting end to end at a 10-degree kink. */
  function continuation(): SceneDescription {
    const scene = curvyScene();
    const kink = (10 * Math.PI) / 180;
    scene.roads = [
      road('main_a', [[-100, 0], [0, 0]], { name: 'Main St' }),
      road('main_b', [[0, 0], [100 * Math.cos(kink), 100 * Math.sin(kink)]], { name: 'Main St' }),
    ];
    return scene;
  }

  it('does not break the centre line where a street merely continues', () => {
    const scene = continuation();
    // Yellow centre-line dashes are 3 m long with 9 m gaps, so any 20 m window
    // around the join must still hold paint if the line was not cut.
    const near = tris(scene, 'lane-markings').filter((t) =>
      t.p.every(([x, y]) => Math.hypot(x, y) < 10),
    );
    expect(near.length).toBeGreaterThan(0);
    // And no stop bar: every marking there is a thin longitudinal dash.
    for (const t of near) {
      const ys = t.p.map((p) => p[1]);
      expect(Math.max(...ys) - Math.min(...ys)).toBeLessThan(1.0);
    }
  });

  it('keeps stop bars off the cross street at a skewed junction', () => {
    const scene = curvyScene();
    const skew = (30 * Math.PI) / 180;
    scene.roads = [
      road('through', [[-100, 0], [100, 0]], { lanes_forward: 2, lanes_backward: 2 }),
      road('skewed', [
        [-100 * Math.cos(skew), -100 * Math.sin(skew)],
        [100 * Math.cos(skew), 100 * Math.sin(skew)],
      ]),
    ];
    const bad = tris(scene, 'lane-markings').filter((t) => {
      const cx = (t.p[0][0] + t.p[1][0] + t.p[2][0]) / 3;
      const cy = (t.p[0][1] + t.p[1][1] + t.p[2][1]) / 3;
      // Inside both carriageways at once = paint in the junction box.
      return scene.roads.every((r) => distToLine(r.centerline, cx, cy) < half(r));
    });
    expect(bad).toHaveLength(0);
  });
});

describe('pavement follows the kerb round a bend', () => {
  it('leaves no wedge gap on the outside of a bend', () => {
    const scene = curvyScene();
    const r = scene.roads.find((x) => x.id === 'curve_local')!;
    const walk = tris(scene, 'sidewalks');
    const pts = r.centerline;
    const missing: Vec2[] = [];
    for (let k = 1; k < pts.length - 1; k++) {
      // Radially outward from the arc's centre (the origin) through the vertex:
      // the bisector of the corner, where square-ended slabs leave their gap.
      const len = Math.hypot(pts[k][0], pts[k][1]);
      const d = half(r) + 1.4;
      const p: Vec2 = [pts[k][0] + (pts[k][0] / len) * d, pts[k][1] + (pts[k][1] / len) * d];
      if (!walk.some((t) => inTri(p[0], p[1], t.p))) missing.push(p);
    }
    expect(missing).toHaveLength(0);
  });
});

describe('stop bars', () => {
  /** A four-way crossroads of two-lane streets, optionally with devices. */
  function crossroads(signs: SceneDescription['stop_signs']): SceneDescription {
    const scene = curvyScene();
    scene.roads = [
      road('ew', [[-100, 0], [100, 0]]),
      road('ns', [[0, -100], [0, 100]]),
    ];
    scene.stop_signs = signs;
    return scene;
  }

  /** Thick transverse white bars: across an east-west road they are tall in y. */
  const bars = (scene: SceneDescription) =>
    tris(scene, 'lane-markings').filter((t) => {
      const xs = t.p.map((p) => p[0]);
      const ys = t.p.map((p) => p[1]);
      const w = Math.max(...xs) - Math.min(...xs);
      const h = Math.max(...ys) - Math.min(...ys);
      // Stop bars are 0.5 m deep and a lane wide; lane lines are 0.12 m wide.
      return (w > 0.3 && h > 2) || (h > 0.3 && w > 2);
    });

  it('paints none at a junction nothing controls', () => {
    expect(bars(crossroads([]))).toHaveLength(0);
  });

  it('paints one per controlled approach, on that approach', () => {
    // Eastbound traffic on `ew` arrives from the west; its sign faces west
    // (heading pi) and stands on the right-hand (south) kerb before the junction.
    const scene = crossroads([{ id: 'ss_eb', position: [-9, -5], heading: Math.PI }]);
    const found = bars(scene);
    expect(found.length).toBeGreaterThan(0);
    for (const t of found) {
      for (const [x, y] of t.p) {
        expect(x).toBeLessThan(0); // west of the junction
        expect(y).toBeLessThan(0.01); // across the eastbound (south) lane only
      }
    }
  });
});

describe('on real terrain, everything sits on the ground', () => {
  const hilly = (): SceneDescription =>
    JSON.parse(readFileSync(resolve(__dirname, 'fixtures/nobHillScene.json'), 'utf8'));

  it('is actually hilly', () => {
    const field = TerrainField.from(hilly().terrain);
    expect(field.flat).toBe(false);
    expect(field.max - field.min).toBeGreaterThan(50);
  });

  it('drapes tarmac, paint and pavement at their stack height above the ground', () => {
    const scene = hilly();
    const field = TerrainField.from(scene.terrain);
    const bands: Array<[string, number, number]> = [
      ['roads', 0.019, 0.025],
      ['lane-markings', 0.049, 0.051],
      ['crosswalks', 0.054, 0.056],
      // Pavement tops at 0.16; its kerb face runs down to road level.
      ['sidewalks', 0.019, 0.161],
    ];
    for (const [name, lo, hi] of bands) {
      const off = tris(scene, name).flatMap((t) =>
        t.p.map(([x, y], k) => t.h[k] - field.heightAt(x, y)),
      );
      expect(off.length, name).toBeGreaterThan(0);
      expect(Math.min(...off), name).toBeGreaterThan(lo - 1e-3);
      expect(Math.max(...off), name).toBeLessThan(hi + 1e-3);
    }
  });

  it('never lets the ground mesh rise through a road', () => {
    const scene = hilly();
    const field = TerrainField.from(scene.terrain);
    const world = buildWorld(scene);
    const mesh = world.root.getObjectByName('terrain') as THREE.Mesh;
    const pos = mesh.geometry.getAttribute('position');
    const { cols, cell, originX, originY } = field;
    /** Height of the drawn ground mesh (not the field) at (x, y). */
    const meshAt = (x: number, y: number) => {
      const fx = (x - originX) / cell;
      const fy = (y - originY) / cell;
      const c = Math.floor(fx);
      const r = Math.floor(fy);
      const u = fx - c;
      const v = fy - r;
      const h = (rr: number, cc: number) => pos.getY(rr * cols + cc);
      // Triangles (a, b, e) and (a, e, d): a=(r,c) b=(r,c+1) d=(r+1,c) e=(r+1,c+1).
      return u >= v
        ? h(r, c) + u * (h(r, c + 1) - h(r, c)) + v * (h(r + 1, c + 1) - h(r, c + 1))
        : h(r, c) + v * (h(r + 1, c) - h(r, c)) + u * (h(r + 1, c + 1) - h(r + 1, c));
    };
    let worst = -Infinity;
    for (const r of scene.roads) {
      for (const [x, y] of samples(r.centerline, 0, 2)) {
        if (x < originX || y < originY) continue;
        if (x >= originX + (field.cols - 1) * cell || y >= originY + (field.rows - 1) * cell) continue;
        worst = Math.max(worst, meshAt(x, y) - (field.heightAt(x, y) + 0.02));
      }
    }
    expect(worst).toBeLessThan(0);
  });

  it('stands every tree, sign post and signal pole on the ground under it', () => {
    const scene = hilly();
    const field = TerrainField.from(scene.terrain);
    const world = buildWorld(scene);
    const m = new THREE.Matrix4();
    const p = new THREE.Vector3();
    const check = (group: string, meshIndex: number, positions: Vec2[]) => {
      const inst = (world.root.getObjectByName(group) as THREE.Group).children[meshIndex] as THREE.InstancedMesh;
      positions.forEach((pos, i) => {
        inst.getMatrixAt(i, m);
        p.setFromMatrixPosition(m);
        expect(p.y, `${group} ${i}`).toBeCloseTo(field.heightAt(pos[0], pos[1]), 3);
      });
    };
    expect(scene.trees.length).toBeGreaterThan(0);
    check('trees', 0, scene.trees.map((t) => t.position));
    if (scene.stop_signs.length) check('stop-signs', 0, scene.stop_signs.map((s) => s.position));
    if (scene.traffic_lights.length) check('traffic-lights', 0, scene.traffic_lights.map((s) => s.position));
  });

  it('never leaves a gap under a building on a slope', () => {
    const scene = hilly();
    const field = TerrainField.from(scene.terrain);
    const world = buildWorld(scene);
    const pos = (world.root.getObjectByName('buildings') as THREE.Mesh).geometry.getAttribute('position');
    let lowest = Infinity;
    for (let i = 0; i < pos.count; i++) lowest = Math.min(lowest, pos.getY(i));
    const groundUnder = scene.buildings.flatMap((b) => b.footprint.map(([x, y]) => field.heightAt(x, y)));
    // The merged mesh's lowest wall starts below the lowest ground under ANY
    // building; and per building, below each of its own corners (checked by
    // construction in world.ts, pinned here on the whole set).
    expect(lowest).toBeLessThan(Math.min(...groundUnder));
  });
});

describe('vehicles ride the slope', () => {
  it('pitches nose-up climbing and rolls toward the low side', () => {
    // Ground rising 10% to the east and 5% to the north.
    const ground = (x: number, y: number) => 0.1 * x + 0.05 * y;
    const east = attitudeOn(ground, 0, 0, 0, 4.6, 1.9);
    expect(east.pitch).toBeCloseTo(Math.atan(0.1), 3);
    // Facing east, north is to the LEFT and higher: the right side is low.
    expect(east.roll).toBeCloseTo(Math.atan(0.05), 3);
    expect(east.y).toBeCloseTo(0, 6);
    const west = attitudeOn(ground, 0, 0, Math.PI, 4.6, 1.9);
    expect(west.pitch).toBeCloseTo(-Math.atan(0.1), 3);
    expect(west.roll).toBeCloseTo(-Math.atan(0.05), 3);
  });
});
