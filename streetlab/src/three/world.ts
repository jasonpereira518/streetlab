/**
 * Turns a `SceneDescription` into renderable geometry.
 *
 * Draw-call budget is the design constraint here. Everything that shares a
 * material is merged into one buffer (road surface, kerbs, markings,
 * crosswalks, every building in the city), and everything that repeats a shape
 * is an InstancedMesh (trees, signal poles, lamps, stop signs). A 3x3 grid with
 * ~30 buildings, ~250 trees and 8 signal heads lands around 30 draw calls
 * before shadows.
 */
import * as THREE from 'three/webgpu';
import { attribute, mix, normalLocal, smoothstep, uv, float } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type {
  LayerKey,
  Road,
  SceneDescription,
  SignalPhase,
  SignalState,
  Vec2,
} from '../schema';
import { signalColor } from '../ui/theme';
import {
  MeshBuilder,
  Polyline,
  dashRuns,
  stripe,
  subtractIntervals,
  worldToThree,
} from './meshBuilder';
import type { Interval } from './meshBuilder';
import { speedLimitTexture, stopFaceTexture, streetNameTexture } from './labels';

/* ------------------------------------------------------------------ */
/* Palette                                                             */
/* ------------------------------------------------------------------ */

const C = {
  asphalt: new THREE.Color('#7E8894'),
  asphaltArterial: new THREE.Color('#78828E'),
  sidewalk: new THREE.Color('#D5DAE1'),
  kerb: new THREE.Color('#BFC6CF'),
  markWhite: new THREE.Color('#F2F5F8'),
  markYellow: new THREE.Color('#E3BE4A'),
  crosswalk: new THREE.Color('#EDF1F5'),
  trunk: new THREE.Color('#8A7259'),
  pole: new THREE.Color('#4A525C'),
  signalBody: new THREE.Color('#2E353D'),
  stopRed: new THREE.Color('#C0392B'),
  signGreen: new THREE.Color('#0B6B45'),
};

const CANOPY_GREENS = ['#7FA867', '#6E9A5C', '#8CB575', '#5F8E52'];

const SIDEWALK_W = 2.8;
const SIDEWALK_H = 0.16;

/** Height stack, kept in one place so nothing z-fights. */
const Y = {
  road: 0.02,
  marking: 0.05,
  crosswalk: 0.055,
  sidewalk: SIDEWALK_H,
};

/* ------------------------------------------------------------------ */
/* Public surface                                                      */
/* ------------------------------------------------------------------ */

export interface World {
  root: THREE.Group;
  /** Drive signal lamps from the live frame. */
  updateSignals(states: SignalState[], time: number): void;
  setLayerVisible(layer: LayerKey, visible: boolean): void;
  dispose(): void;
}

const carriagewayHalfWidth = (r: Road): number =>
  ((r.lanes_forward + r.lanes_backward) * r.lane_width_m) / 2;

/* ------------------------------------------------------------------ */
/* Intersections                                                       */
/* ------------------------------------------------------------------ */

interface Crossing {
  /** Arc length along the road being cut. */
  s: number;
  /** Half-width of the crossing road. */
  half: number;
  /** Crossing point in world coordinates. */
  at: Vec2;
}

/** Parametric segment-segment intersection; null when parallel or disjoint. */
function segIntersect(
  a0: Vec2,
  a1: Vec2,
  b0: Vec2,
  b1: Vec2,
): { ta: number; tb: number } | null {
  const rx = a1[0] - a0[0];
  const ry = a1[1] - a0[1];
  const sx = b1[0] - b0[0];
  const sy = b1[1] - b0[1];
  const denom = rx * sy - ry * sx;
  if (Math.abs(denom) < 1e-9) return null;
  const qpx = b0[0] - a0[0];
  const qpy = b0[1] - a0[1];
  const ta = (qpx * sy - qpy * sx) / denom;
  const tb = (qpx * ry - qpy * rx) / denom;
  if (ta < 0 || ta > 1 || tb < 0 || tb > 1) return null;
  return { ta, tb };
}

function findCrossings(roads: Road[], lines: Polyline[]): Crossing[][] {
  const out: Crossing[][] = roads.map(() => []);
  for (let i = 0; i < roads.length; i++) {
    for (let j = 0; j < roads.length; j++) {
      if (i === j) continue;
      const li = lines[i];
      const lj = lines[j];
      for (let si = 1; si < li.points.length; si++) {
        for (let sj = 1; sj < lj.points.length; sj++) {
          const hit = segIntersect(
            li.points[si - 1],
            li.points[si],
            lj.points[sj - 1],
            lj.points[sj],
          );
          if (!hit) continue;
          const segLen = li.cum[si] - li.cum[si - 1];
          const s = li.cum[si - 1] + hit.ta * segLen;
          out[i].push({
            s,
            half: carriagewayHalfWidth(roads[j]),
            at: [
              li.points[si - 1][0] +
                hit.ta * (li.points[si][0] - li.points[si - 1][0]),
              li.points[si - 1][1] +
                hit.ta * (li.points[si][1] - li.points[si - 1][1]),
            ],
          });
        }
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Build                                                               */
/* ------------------------------------------------------------------ */

export function buildWorld(scene: SceneDescription): World {
  const root = new THREE.Group();
  root.name = 'world';

  const disposables: Array<{ dispose(): void }> = [];
  const layerNodes = new Map<LayerKey, THREE.Object3D[]>();
  const track = (layer: LayerKey, obj: THREE.Object3D) => {
    const list = layerNodes.get(layer) ?? [];
    list.push(obj);
    layerNodes.set(layer, list);
  };

  const lines = scene.roads.map((r) => new Polyline(r.centerline));
  const crossings = findCrossings(scene.roads, lines);

  /* -------- road surface, kerbs and sidewalks -------- */

  const surface = new MeshBuilder();
  const paving = new MeshBuilder();

  scene.roads.forEach((road, i) => {
    const line = lines[i];
    const half = carriagewayHalfWidth(road);
    // A hair of vertical separation per road guarantees a stable draw order
    // where carriageways overlap at intersections.
    const yRoad = Y.road + i * 0.0009;
    const col = road.road_class === 'arterial' ? C.asphaltArterial : C.asphalt;

    stripe(surface, line, { from: 0, to: line.length }, 0, half * 2, yRoad, col);

    if (!road.has_sidewalk) return;
    // Sidewalks stop short of every crossing carriageway.
    const holes: Interval[] = crossings[i].map((c) => ({
      from: c.s - c.half - SIDEWALK_W,
      to: c.s + c.half + SIDEWALK_W,
    }));
    for (const span of subtractIntervals(line.length, holes)) {
      for (const side of [1, -1]) {
        const inner = side * (half + 0.02);
        const outer = side * (half + SIDEWALK_W);
        const mid = (inner + outer) / 2;
        stripe(
          paving,
          line,
          span,
          mid,
          SIDEWALK_W,
          Y.sidewalk,
          C.sidewalk,
        );
        // Kerb face, so the pavement reads as raised rather than painted.
        kerbFace(paving, line, span, inner, side, yRoad);
      }
    }
  });

  // Sidewalk corner patches so pavements meet at intersections.
  const cornerDone = new Set<string>();
  scene.roads.forEach((road, i) => {
    for (const c of crossings[i]) {
      const key = `${c.at[0].toFixed(2)}:${c.at[1].toFixed(2)}`;
      if (cornerDone.has(key)) continue;
      cornerDone.add(key);
      const hx = carriagewayHalfWidth(road);
      const hy = c.half;
      for (const sx of [-1, 1]) {
        for (const sy of [-1, 1]) {
          const x0 = c.at[0] + sx * hx;
          const y0 = c.at[1] + sy * hy;
          const x1 = c.at[0] + sx * (hx + SIDEWALK_W);
          const y1 = c.at[1] + sy * (hy + SIDEWALK_W);
          paving.box(
            Math.min(x0, x1),
            Math.min(y0, y1),
            Math.max(x0, x1),
            Math.max(y0, y1),
            0,
            Y.sidewalk,
            C.sidewalk,
          );
        }
      }
    }
  });

  const surfaceMat = surfaceMaterial();
  const roadMesh = new THREE.Mesh(surface.build(), surfaceMat);
  roadMesh.receiveShadow = true;
  roadMesh.name = 'roads';
  root.add(roadMesh);

  const paveMesh = new THREE.Mesh(paving.build(), surfaceMat);
  paveMesh.receiveShadow = true;
  paveMesh.castShadow = false;
  paveMesh.name = 'sidewalks';
  root.add(paveMesh);
  disposables.push(roadMesh.geometry, paveMesh.geometry, surfaceMat);

  /* -------- lane markings -------- */

  const markings = new MeshBuilder();
  scene.roads.forEach((road, i) => {
    const line = lines[i];
    const half = carriagewayHalfWidth(road);
    const holes: Interval[] = crossings[i].map((c) => ({
      from: c.s - c.half - 1.5,
      to: c.s + c.half + 1.5,
    }));
    const spans = subtractIntervals(line.length, holes);

    for (const span of spans) {
      // Centre divider.
      if (road.center_marking === 'double_yellow') {
        for (const o of [0.16, -0.16]) {
          stripe(markings, line, span, o, 0.13, Y.marking, C.markYellow);
        }
      } else if (road.center_marking === 'solid_white') {
        stripe(markings, line, span, 0, 0.14, Y.marking, C.markWhite);
      }

      // Dashed dividers between same-direction lanes.
      for (const dir of [1, -1] as const) {
        const count = dir === 1 ? road.lanes_forward : road.lanes_backward;
        for (let k = 1; k < count; k++) {
          const lateral = dir * k * road.lane_width_m;
          for (const run of dashRuns(span, 3, 4.4)) {
            stripe(markings, line, run, lateral, 0.12, Y.marking, C.markWhite);
          }
        }
      }

      // Edge line just inside the kerb.
      for (const side of [1, -1]) {
        stripe(
          markings,
          line,
          span,
          side * (half - 0.28),
          0.12,
          Y.marking,
          C.markWhite,
        );
      }

      // Stop bar on the approach to each intersection.
      for (const c of crossings[i]) {
        for (const dir of [1, -1] as const) {
          const at = c.s - dir * (c.half + 2.2);
          if (at < span.from || at > span.to) continue;
          const laneSpan = dir === 1 ? road.lanes_forward : road.lanes_backward;
          if (laneSpan === 0) continue;
          const mid = (dir * laneSpan * road.lane_width_m) / 2;
          stripe(
            markings,
            line,
            { from: at - 0.24, to: at + 0.24 },
            mid,
            laneSpan * road.lane_width_m - 0.2,
            Y.marking,
            C.markWhite,
          );
        }
      }
    }
  });

  const markMat = flatMaterial();
  const markMesh = new THREE.Mesh(markings.build(), markMat);
  markMesh.name = 'lane-markings';
  root.add(markMesh);
  track('lane_markings', markMesh);
  disposables.push(markMesh.geometry, markMat);

  /* -------- crosswalks -------- */

  const walks = new MeshBuilder();
  for (const xw of scene.crosswalks) {
    const [cx, cy] = xw.center;
    const dx = Math.cos(xw.heading);
    const dy = Math.sin(xw.heading);
    // Perpendicular: the direction the striped band is thick in.
    const px = -dy;
    const py = dx;
    const bars = Math.max(2, Math.round(xw.length_m / 1.1));
    const barW = (xw.length_m / bars) * 0.58;
    for (let b = 0; b < bars; b++) {
      const t = (b + 0.5) / bars - 0.5;
      const along = t * xw.length_m;
      const corners: [Vec2, Vec2, Vec2, Vec2] = [
        [
          cx + dx * (along - barW / 2) + px * (xw.width_m / 2),
          cy + dy * (along - barW / 2) + py * (xw.width_m / 2),
        ],
        [
          cx + dx * (along + barW / 2) + px * (xw.width_m / 2),
          cy + dy * (along + barW / 2) + py * (xw.width_m / 2),
        ],
        [
          cx + dx * (along + barW / 2) - px * (xw.width_m / 2),
          cy + dy * (along + barW / 2) - py * (xw.width_m / 2),
        ],
        [
          cx + dx * (along - barW / 2) - px * (xw.width_m / 2),
          cy + dy * (along - barW / 2) - py * (xw.width_m / 2),
        ],
      ];
      walks.flatQuad(corners, Y.crosswalk, C.crosswalk);
    }
  }
  const walkMat = flatMaterial();
  const walkMesh = new THREE.Mesh(walks.build(), walkMat);
  walkMesh.name = 'crosswalks';
  root.add(walkMesh);
  track('crosswalks', walkMesh);
  disposables.push(walkMesh.geometry, walkMat);

  /* -------- buildings -------- */

  if (scene.buildings.length) {
    const geos: THREE.BufferGeometry[] = [];
    const facade = new THREE.Color();
    const roof = new THREE.Color();
    for (const b of scene.buildings) {
      const shape = new THREE.Shape(
        b.footprint.map(([x, y]) => new THREE.Vector2(x, y)),
      );
      const geo = new THREE.ExtrudeGeometry(shape, {
        depth: b.height_m,
        bevelEnabled: false,
        curveSegments: 1,
      });
      // Shape XY -> world XY, extrusion Z -> height.
      geo.rotateX(-Math.PI / 2);

      facade.set(b.color).convertSRGBToLinear();
      roof.set(b.roof_color).convertSRGBToLinear();
      const count = geo.attributes.position.count;
      const colors = new Float32Array(count * 3);
      // Group 0 is the cap pair, group 1 the extruded walls.
      for (const g of geo.groups) {
        const c = g.materialIndex === 0 ? roof : facade;
        for (let v = g.start; v < g.start + g.count; v++) {
          colors[v * 3] = c.r;
          colors[v * 3 + 1] = c.g;
          colors[v * 3 + 2] = c.b;
        }
      }
      geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
      geo.clearGroups();
      geo.deleteAttribute('uv2');
      geos.push(geo);
    }
    const merged = mergeGeometries(geos, false);
    for (const g of geos) g.dispose();
    if (merged) {
      const mat = buildingMaterial();
      const mesh = new THREE.Mesh(merged, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.name = 'buildings';
      root.add(mesh);
      track('buildings', mesh);
      disposables.push(merged, mat);
    }
  }

  /* -------- trees -------- */

  if (scene.trees.length) {
    const n = scene.trees.length;
    const trunkGeo = new THREE.CylinderGeometry(0.7, 1, 1, 6, 1);
    trunkGeo.translate(0, 0.5, 0);
    const canopyGeo = new THREE.IcosahedronGeometry(1, 1);

    const trunkMat = foliageMaterial(0.95);
    const canopyMat = foliageMaterial(0.85);
    const trunks = new THREE.InstancedMesh(trunkGeo, trunkMat, n);
    const canopies = new THREE.InstancedMesh(canopyGeo, canopyMat, n);
    trunks.castShadow = true;
    canopies.castShadow = true;
    canopies.receiveShadow = true;

    const m = new THREE.Matrix4();
    const q = new THREE.Quaternion();
    const pos = new THREE.Vector3();
    const scale = new THREE.Vector3();
    const col = new THREE.Color();

    scene.trees.forEach((t, i) => {
      const [tx, tz] = worldToThreeXZ(t.position);
      const trunkH = t.height_m * 0.46;
      pos.set(tx, 0, tz);
      scale.set(t.trunk_radius_m, trunkH, t.trunk_radius_m);
      q.identity();
      trunks.setMatrixAt(i, m.compose(pos, q, scale));
      trunks.setColorAt(i, col.set(C.trunk));

      const wobble = 0.82 + t.variant * 0.36;
      pos.set(tx, trunkH + t.canopy_radius_m * 0.72, tz);
      scale.set(
        t.canopy_radius_m * wobble,
        t.canopy_radius_m * (1.05 + t.variant * 0.35),
        t.canopy_radius_m * (1.9 - wobble),
      );
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), t.variant * Math.PI * 2);
      canopies.setMatrixAt(i, m.compose(pos, q, scale));
      canopies.setColorAt(
        i,
        col.set(CANOPY_GREENS[Math.floor(t.variant * CANOPY_GREENS.length) % CANOPY_GREENS.length]),
      );
    });
    trunks.instanceMatrix.needsUpdate = true;
    canopies.instanceMatrix.needsUpdate = true;
    if (trunks.instanceColor) trunks.instanceColor.needsUpdate = true;
    if (canopies.instanceColor) canopies.instanceColor.needsUpdate = true;

    const group = new THREE.Group();
    group.name = 'trees';
    group.add(trunks, canopies);
    root.add(group);
    track('trees', group);
    disposables.push(trunkGeo, canopyGeo, trunkMat, canopyMat);
  }

  /* -------- traffic lights -------- */

  const signals = buildTrafficLights(scene, disposables);
  if (signals) {
    root.add(signals.group);
    track('traffic_lights', signals.group);
  }

  /* -------- stop signs -------- */

  if (scene.stop_signs.length) {
    const group = buildStopSigns(scene, disposables);
    root.add(group);
    track('traffic_lights', group);
  }

  /* -------- street name signs -------- */

  if (scene.street_signs.length) {
    const group = buildStreetSigns(scene, disposables);
    root.add(group);
    track('labels', group);
  }

  return {
    root,
    updateSignals(states, time) {
      signals?.update(states, time);
    },
    setLayerVisible(layer, visible) {
      for (const node of layerNodes.get(layer) ?? []) node.visible = visible;
    },
    dispose() {
      for (const d of disposables) d.dispose();
      root.clear();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Sub-builders                                                        */
/* ------------------------------------------------------------------ */

const worldToThreeXZ = (p: Vec2): [number, number] => [p[0], -p[1]];

/** Vertical kerb face along one side of a road span. */
function kerbFace(
  builder: MeshBuilder,
  line: Polyline,
  span: Interval,
  lateral: number,
  _side: number,
  roadY: number,
): void {
  const STEP = 6;
  const n = Math.max(1, Math.ceil((span.to - span.from) / STEP));
  for (let i = 0; i < n; i++) {
    const s0 = span.from + ((span.to - span.from) * i) / n;
    const s1 = span.from + ((span.to - span.from) * (i + 1)) / n;
    const a = line.offsetAt(s0, lateral);
    const b = line.offsetAt(s1, lateral);
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy) || 1;
    const normal: [number, number, number] = [-dy / len, 0, -dx / len];
    builder.quad(
      worldToThree(a[0], a[1], roadY),
      worldToThree(b[0], b[1], roadY),
      worldToThree(b[0], b[1], Y.sidewalk),
      worldToThree(a[0], a[1], Y.sidewalk),
      normal,
      C.kerb,
    );
  }
}

interface SignalRig {
  group: THREE.Group;
  update(states: SignalState[], time: number): void;
}

function buildTrafficLights(
  scene: SceneDescription,
  disposables: Array<{ dispose(): void }>,
): SignalRig | null {
  const lights = scene.traffic_lights;
  if (!lights.length) return null;
  const n = lights.length;

  const poleGeo = new THREE.CylinderGeometry(0.085, 0.11, 1, 8);
  poleGeo.translate(0, 0.5, 0);
  const armGeo = new THREE.BoxGeometry(1, 0.1, 0.1);
  armGeo.translate(0.5, 0, 0);
  const bodyGeo = new THREE.BoxGeometry(0.24, 1.02, 0.32);
  const lensGeo = new THREE.CylinderGeometry(0.115, 0.115, 0.07, 12);
  lensGeo.rotateZ(Math.PI / 2);

  const metal = metalMaterial(C.pole);
  const bodyMat = metalMaterial(C.signalBody);
  const poles = new THREE.InstancedMesh(poleGeo, metal, n);
  const arms = new THREE.InstancedMesh(armGeo, metal, n);
  const bodies = new THREE.InstancedMesh(bodyGeo, bodyMat, n);
  poles.castShadow = true;
  arms.castShadow = true;
  bodies.castShadow = true;

  const PHASES = ['red', 'yellow', 'green'] as const;
  const lamps = PHASES.map((p) => {
    const mat = lampMaterial(signalColor[p]);
    disposables.push(mat);
    const mesh = new THREE.InstancedMesh(lensGeo, mat, n);
    mesh.frustumCulled = false;
    return mesh;
  });
  // Unlit lenses, always present so a dark head still reads as a signal.
  const darkMat = metalMaterial(new THREE.Color('#1D2229'));
  const darkLenses = new THREE.InstancedMesh(lensGeo, darkMat, n * 3);

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const scale = new THREE.Vector3();
  const up = new THREE.Vector3(0, 1, 0);

  /** Head position per light, cached for the per-frame lamp update. */
  const heads: Array<{ x: number; y: number; z: number; heading: number }> = [];

  lights.forEach((tl, i) => {
    const [px, pz] = worldToThreeXZ(tl.position);
    // Mast arm reaches out along `heading` rotated -90 degrees.
    const armWx = Math.sin(tl.heading);
    const armWy = -Math.cos(tl.heading);
    const [ax, az] = worldToThreeXZ([armWx, armWy]);

    q.identity();
    poles.setMatrixAt(
      i,
      m.compose(pos.set(px, 0, pz), q, scale.set(1, tl.height_m, 1)),
    );

    const armAngle = Math.atan2(-az, ax);
    q.setFromAxisAngle(up, armAngle);
    arms.setMatrixAt(
      i,
      m.compose(
        pos.set(px, tl.height_m - 0.12, pz),
        q,
        scale.set(Math.max(0.001, tl.mast_arm_m), 1, 1),
      ),
    );

    const hx = px + ax * tl.mast_arm_m;
    const hz = pz + az * tl.mast_arm_m;
    const hy = tl.height_m - 0.72;
    heads.push({ x: hx, y: hy, z: hz, heading: tl.heading });

    q.setFromAxisAngle(up, tl.heading);
    bodies.setMatrixAt(i, m.compose(pos.set(hx, hy, hz), q, scale.set(1, 1, 1)));

    // Dark lenses on the facing side of the housing.
    const fx = Math.cos(tl.heading);
    const fz = -Math.sin(tl.heading);
    PHASES.forEach((_p, k) => {
      const dy = 0.33 - k * 0.33;
      darkLenses.setMatrixAt(
        i * 3 + k,
        m.compose(
          pos.set(hx + fx * 0.14, hy + dy, hz + fz * 0.14),
          q,
          scale.set(1, 1, 1),
        ),
      );
    });
  });

  poles.instanceMatrix.needsUpdate = true;
  arms.instanceMatrix.needsUpdate = true;
  bodies.instanceMatrix.needsUpdate = true;
  darkLenses.instanceMatrix.needsUpdate = true;

  const group = new THREE.Group();
  group.name = 'traffic-lights';
  group.add(poles, arms, bodies, darkLenses, ...lamps);
  disposables.push(
    poleGeo,
    armGeo,
    bodyGeo,
    lensGeo,
    metal,
    bodyMat,
    darkMat,
  );

  const hidden = new THREE.Vector3(0, 0, 0);
  const lastKey: string[] = new Array(n).fill('');

  const update = (states: SignalState[], time: number) => {
    const byId = new Map(states.map((s) => [s.id, s]));
    let dirty = false;
    lights.forEach((tl, i) => {
      const st = byId.get(tl.id);
      const phase: SignalPhase = st?.phase ?? 'off';
      // Flashing yellow blinks at 1 Hz; everything else is steady.
      const blink = phase === 'flashing_yellow' && Math.floor(time * 2) % 2 === 0;
      const key = `${phase}${blink ? '1' : '0'}`;
      if (lastKey[i] === key) return;
      lastKey[i] = key;
      dirty = true;

      const head = heads[i];
      const fx = Math.cos(head.heading);
      const fz = -Math.sin(head.heading);
      q.setFromAxisAngle(up, head.heading);

      PHASES.forEach((p, k) => {
        const on =
          phase === 'flashing_yellow'
            ? p === 'yellow' && !blink
            : phase === p;
        const dy = 0.33 - k * 0.33;
        if (on) {
          lamps[k].setMatrixAt(
            i,
            m.compose(
              pos.set(head.x + fx * 0.17, head.y + dy, head.z + fz * 0.17),
              q,
              scale.set(1.12, 1.12, 1.12),
            ),
          );
        } else {
          // Zero scale collapses the instance to degenerate triangles, which
          // the rasteriser drops. Note `Matrix4.decompose` cannot read this
          // back — a zero determinant makes it report a scale of 1 — so
          // inspect the matrix elements directly if you need to assert on it.
          lamps[k].setMatrixAt(
            i,
            m.compose(hidden, q, scale.set(0, 0, 0)),
          );
        }
      });
    });
    if (dirty) for (const l of lamps) l.instanceMatrix.needsUpdate = true;
  };

  // Start dark so the first real frame lights them.
  update([], 0);

  return { group, update };
}

function buildStopSigns(
  scene: SceneDescription,
  disposables: Array<{ dispose(): void }>,
): THREE.Group {
  const n = scene.stop_signs.length;
  const postGeo = new THREE.CylinderGeometry(0.045, 0.05, 1, 6);
  postGeo.translate(0, 0.5, 0);
  const octGeo = new THREE.CylinderGeometry(0.42, 0.42, 0.05, 8);
  octGeo.rotateZ(Math.PI / 2);
  octGeo.rotateX(Math.PI / 8);
  const faceGeo = new THREE.PlaneGeometry(0.78, 0.78);
  faceGeo.rotateY(Math.PI / 2);

  const postMat = metalMaterial(C.pole);
  const octMat = metalMaterial(C.stopRed);
  const faceTex = stopFaceTexture();
  const faceMat = new THREE.MeshBasicNodeMaterial({
    map: faceTex,
    transparent: true,
    depthWrite: false,
  });

  const posts = new THREE.InstancedMesh(postGeo, postMat, n);
  const octs = new THREE.InstancedMesh(octGeo, octMat, n);
  const faces = new THREE.InstancedMesh(faceGeo, faceMat, n);
  posts.castShadow = true;
  octs.castShadow = true;

  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const pos = new THREE.Vector3();
  const one = new THREE.Vector3(1, 1, 1);
  const up = new THREE.Vector3(0, 1, 0);

  scene.stop_signs.forEach((s, i) => {
    const [x, z] = worldToThreeXZ(s.position);
    q.identity();
    posts.setMatrixAt(i, m.compose(pos.set(x, 0, z), q, new THREE.Vector3(1, 2.2, 1)));
    q.setFromAxisAngle(up, s.heading);
    octs.setMatrixAt(i, m.compose(pos.set(x, 2.16, z), q, one));
    const fx = Math.cos(s.heading) * 0.028;
    const fz = -Math.sin(s.heading) * 0.028;
    faces.setMatrixAt(i, m.compose(pos.set(x + fx, 2.16, z + fz), q, one));
  });
  posts.instanceMatrix.needsUpdate = true;
  octs.instanceMatrix.needsUpdate = true;
  faces.instanceMatrix.needsUpdate = true;

  const group = new THREE.Group();
  group.name = 'stop-signs';
  group.add(posts, octs, faces);
  disposables.push(postGeo, octGeo, faceGeo, postMat, octMat, faceMat, faceTex);
  return group;
}

function buildStreetSigns(
  scene: SceneDescription,
  disposables: Array<{ dispose(): void }>,
): THREE.Group {
  const group = new THREE.Group();
  group.name = 'street-signs';

  // One instanced post per sign position (blades share a post).
  const seen = new Map<string, number>();
  const posts: Array<[number, number]> = [];
  for (const s of scene.street_signs) {
    const key = `${s.position[0].toFixed(2)}:${s.position[1].toFixed(2)}`;
    if (seen.has(key)) continue;
    seen.set(key, posts.length);
    posts.push(worldToThreeXZ(s.position));
  }

  const postGeo = new THREE.CylinderGeometry(0.04, 0.045, 1, 6);
  postGeo.translate(0, 0.5, 0);
  const postMat = metalMaterial(C.pole);
  const postMesh = new THREE.InstancedMesh(postGeo, postMat, posts.length);
  postMesh.castShadow = true;
  const m = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  posts.forEach(([x, z], i) => {
    postMesh.setMatrixAt(
      i,
      m.compose(new THREE.Vector3(x, 0, z), q.identity(), new THREE.Vector3(1, 3.1, 1)),
    );
  });
  postMesh.instanceMatrix.needsUpdate = true;
  group.add(postMesh);
  disposables.push(postGeo, postMat);

  // Blades carry per-sign text, so each gets its own texture and mesh. With a
  // handful of intersections this stays well inside the draw-call budget.
  const perPost = new Map<string, number>();
  for (const s of scene.street_signs) {
    const key = `${s.position[0].toFixed(2)}:${s.position[1].toFixed(2)}`;
    const nth = perPost.get(key) ?? 0;
    perPost.set(key, nth + 1);
    const [x, z] = worldToThreeXZ(s.position);

    if (s.kind === 'speed_limit') {
      const tex = speedLimitTexture(s.text);
      const mat = new THREE.MeshBasicNodeMaterial({
        map: tex,
        side: THREE.DoubleSide,
      });
      const geo = new THREE.PlaneGeometry(0.62, 0.93);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x, 2.0, z);
      mesh.rotation.y = s.heading + Math.PI / 2;
      group.add(mesh);
      disposables.push(geo, mat, tex);
      continue;
    }

    const tex = streetNameTexture(s.text);
    const mat = new THREE.MeshBasicNodeMaterial({
      map: tex,
      side: THREE.DoubleSide,
    });
    const geo = new THREE.PlaneGeometry(1.85, 0.46);
    const mesh = new THREE.Mesh(geo, mat);
    // Stack blades down the post so two street names never overlap.
    mesh.position.set(x, 3.0 - nth * 0.56, z);
    // The blade runs parallel to its own carriageway, so the plane's normal is
    // perpendicular to the street's heading.
    mesh.rotation.y = s.heading + Math.PI / 2;
    group.add(mesh);
    disposables.push(geo, mat, tex);
  }

  return group;
}

/* ------------------------------------------------------------------ */
/* Materials (TSL)                                                     */
/* ------------------------------------------------------------------ */

/** Road + pavement: vertex colour with a faint large-scale tonal drift. */
function surfaceMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.94,
    metalness: 0,
  });
  mat.colorNode = attribute<'vec3'>('color', 'vec3');
  return mat;
}

/** Markings and crosswalks: flat, bright, and never shadowed into mud. */
function flatMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.6,
    metalness: 0,
  });
  mat.colorNode = attribute<'vec3'>('color', 'vec3');
  return mat;
}

/**
 * Buildings. Facades get horizontal floor banding and vertical mullions from
 * the extrusion UVs; roofs are left flat. Walls are separated from roofs by
 * their normal, which survives the merge into a single buffer.
 */
function buildingMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({
    roughness: 0.82,
    metalness: 0.02,
  });

  const base = attribute<'vec3'>('color', 'vec3');
  // ExtrudeGeometry side UVs are in world units: x along the wall, y = -height.
  const heightM = uv().y.mul(-1);
  const floorBand = heightM.div(3.4).fract();
  const windows = smoothstep(float(0.12), float(0.3), floorBand).mul(
    smoothstep(float(0.94), float(0.78), floorBand),
  );
  const mullion = smoothstep(
    float(0.06),
    float(0.22),
    uv().x.div(2.1).fract().sub(0.5).abs(),
  );
  const glass = windows.mul(mullion);

  // 1 on walls, 0 on the roof caps.
  const wallness = smoothstep(float(0.72), float(0.28), normalLocal.y.abs());
  const shaded = mix(base, base.mul(0.74), glass.mul(wallness));

  mat.colorNode = shaded;
  return mat;
}

function foliageMaterial(roughness: number): THREE.MeshStandardNodeMaterial {
  return new THREE.MeshStandardNodeMaterial({
    roughness,
    metalness: 0,
    flatShading: true,
  });
}

function metalMaterial(c: THREE.Color): THREE.MeshStandardNodeMaterial {
  return new THREE.MeshStandardNodeMaterial({
    color: c,
    roughness: 0.55,
    metalness: 0.25,
  });
}

/** Lit signal lens: unlit so it reads as emissive under a bright sky. */
function lampMaterial(hex: string): THREE.MeshBasicNodeMaterial {
  return new THREE.MeshBasicNodeMaterial({
    color: new THREE.Color(hex),
    toneMapped: false,
  });
}
