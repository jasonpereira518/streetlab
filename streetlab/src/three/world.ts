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
import type { Interval, P3 } from './meshBuilder';
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

/* ---- stop sign proportions, all in metres ---- */
/** Thickness of the octagonal plate, along the direction it faces. */
const PLATE_T = 0.05;
/** Height of the plate's centre above the pavement. */
const SIGN_Y = 2.16;
/**
 * How far behind the plate the post stands. Must clear half the plate's own
 * thickness plus the post's radius (0.05), or the post shows in front of the
 * sign; the remainder is margin.
 */
const POST_SETBACK = PLATE_T / 2 + 0.05 + 0.015;
/** Post height. Below the plate's top edge (SIGN_Y + 0.42) so it stays hidden. */
const POST_H = 2.45;
/* ---- painted line proportions, all in metres (MUTCD 3A.05/3B) ---- */
/** A normal longitudinal line is 4-6 in wide. */
const LINE_W = 0.12;
/** Broken lines: 10 ft of paint, 30 ft of gap. Exported so tests can pin it. */
export const DASH_M = 3.05;
export const GAP_M = 9.14;
/** Centre-to-centre spacing of the two lines of a double yellow. */
const DOUBLE_GAP_M = 0.2;
/** How far inside the kerb an edge line is painted. */
const EDGE_INSET_M = 0.28;
/** Stop bar depth, along the direction of travel. 12-24 in. */
const STOP_BAR_M = 0.5;

/** How far the printed face floats in front of the plate, to avoid z-fighting. */
const FACE_PROUD = PLATE_T / 2 + 0.01;
/**
 * How far a street-sign blade stands in front of its post. A blade centred on
 * its own post is bisected by it, and since the blade is a zero-thickness
 * plane the post's front half draws straight over the text. Clears the post's
 * 0.045 m radius with margin.
 */
const BLADE_PROUD = 0.06;

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
/* Road surface index                                                  */
/* ------------------------------------------------------------------ */

/** Cell size for the carriageway lookup grid, in metres. */
const CELL_M = 24;
/** Step length along a pavement, in metres. Also its clipping resolution. */
const PAVE_STEP_M = 2.0;
/** Angular divisions of a junction apron. */
const APRON_SEGMENTS = 32;

interface RoadSeg {
  ax: number;
  ay: number;
  bx: number;
  by: number;
  half: number;
}

function pointToSeg(px: number, py: number, s: RoadSeg): number {
  const dx = s.bx - s.ax;
  const dy = s.by - s.ay;
  const L = dx * dx + dy * dy;
  if (L < 1e-12) return Math.hypot(px - s.ax, py - s.ay);
  const t = Math.max(0, Math.min(1, ((px - s.ax) * dx + (py - s.ay) * dy) / L));
  return Math.hypot(px - (s.ax + t * dx), py - (s.ay + t * dy));
}

/**
 * Every carriageway in the scene, bucketed for point lookups.
 *
 * Pavement is generated by offsetting each road on its own, which is only ever
 * correct until another road turns up. This is what lets a strip be clipped
 * against ALL of them at emit time — the one rule that holds whatever angle
 * two streets cross at, and whatever OSM has mapped twice.
 */
class RoadSurfaces {
  private cells = new Map<number, RoadSeg[]>();

  constructor(roads: Road[], lines: Polyline[]) {
    roads.forEach((road, i) => {
      const half = carriagewayHalfWidth(road);
      const pts = lines[i].points;
      for (let k = 1; k < pts.length; k++) {
        const seg: RoadSeg = {
          ax: pts[k - 1][0],
          ay: pts[k - 1][1],
          bx: pts[k][0],
          by: pts[k][1],
          half,
        };
        // Expanded by the half-width, so any point the segment could cover
        // falls in a cell the segment was filed under.
        const x0 = Math.floor((Math.min(seg.ax, seg.bx) - half) / CELL_M);
        const x1 = Math.floor((Math.max(seg.ax, seg.bx) + half) / CELL_M);
        const y0 = Math.floor((Math.min(seg.ay, seg.by) - half) / CELL_M);
        const y1 = Math.floor((Math.max(seg.ay, seg.by) + half) / CELL_M);
        for (let cx = x0; cx <= x1; cx++) {
          for (let cy = y0; cy <= y1; cy++) {
            const key = cx * 100000 + cy;
            const list = this.cells.get(key);
            if (list) list.push(seg);
            else this.cells.set(key, [seg]);
          }
        }
      }
    });
  }

  /** True when the point is on tarmac, beyond `slack` metres of the kerb. */
  covers(x: number, y: number, slack = 0.0): boolean {
    const key = Math.floor(x / CELL_M) * 100000 + Math.floor(y / CELL_M);
    const list = this.cells.get(key);
    if (!list) return false;
    for (const seg of list) {
      if (pointToSeg(x, y, seg) < seg.half - slack) return true;
    }
    return false;
  }

  /** True when any part of a quad lies on tarmac. */
  coversQuad(c: [Vec2, Vec2, Vec2, Vec2]): boolean {
    let mx = 0;
    let my = 0;
    for (const [x, y] of c) {
      if (this.covers(x, y)) return true;
      mx += x / 4;
      my += y / 4;
    }
    if (this.covers(mx, my, 0.05)) return true;
    for (let i = 0; i < 4; i++) {
      const [ax, ay] = c[i];
      const [bx, by] = c[(i + 1) % 4];
      if (this.covers((ax + bx) / 2, (ay + by) / 2)) return true;
    }
    return false;
  }
}

/**
 * Walk one polyline segment in short steps, calling `keep` for each, and hand
 * back the runs of consecutive steps that survived.
 *
 * Stepping decides where the pavement stops; COALESCING is what stops that
 * from costing anything. Emitting a quad per step turned the Nob Hill
 * fixture's pavement from 4,732 triangles into 20,372 — a straight block was
 * being tessellated into fifty slabs that a single quad describes exactly.
 * Runs never span a vertex, so the polyline is straight across every one of
 * them and a merged quad cuts no corners.
 */
function keptRuns(
  length: number,
  step: number,
  keep: (from: number, to: number) => boolean,
): Array<[number, number]> {
  const n = Math.max(1, Math.ceil(length / step));
  const runs: Array<[number, number]> = [];
  let open: number | null = null;
  for (let i = 0; i < n; i++) {
    const a = (length * i) / n;
    const b = (length * (i + 1)) / n;
    if (keep(a, b)) {
      if (open === null) open = a;
    } else if (open !== null) {
      runs.push([open, a]);
      open = null;
    }
  }
  if (open !== null) runs.push([open, length]);
  return runs;
}

/** Each straight segment of a polyline, as origin, unit tangent and length. */
function segmentsOf(line: Polyline): Array<{ ox: number; oy: number; tx: number; ty: number; len: number }> {
  const out = [];
  for (let i = 1; i < line.points.length; i++) {
    const a = line.points[i - 1];
    const b = line.points[i];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (len < 1e-9) continue;
    out.push({ ox: a[0], oy: a[1], tx: (b[0] - a[0]) / len, ty: (b[1] - a[1]) / len, len });
  }
  return out;
}

/**
 * A pavement strip down one side of a road, clipped off wherever it would lie
 * on any carriageway.
 *
 * Offsetting each road on its own is only correct until another road turns up,
 * so where a pavement stops is decided by the tarmac it actually meets — not,
 * as before, by an arc-length interval derived from one crossing road's
 * half-width as though the two streets met at a right angle.
 */
function pavedStrip(
  builder: MeshBuilder,
  line: Polyline,
  inner: number,
  outer: number,
  height: number,
  color: THREE.Color,
  roads: RoadSurfaces,
): void {
  for (const seg of segmentsOf(line)) {
    const nx = -seg.ty;
    const ny = seg.tx;
    const at = (d: number, lat: number): Vec2 => [
      seg.ox + seg.tx * d + nx * lat,
      seg.oy + seg.ty * d + ny * lat,
    ];
    const quad = (a: number, b: number): [Vec2, Vec2, Vec2, Vec2] => [
      at(a, inner),
      at(b, inner),
      at(b, outer),
      at(a, outer),
    ];
    for (const [a, b] of keptRuns(seg.len, PAVE_STEP_M, (f, t) => !roads.coversQuad(quad(f, t)))) {
      builder.flatQuad(quad(a, b), height, color);
    }
  }
}

/**
 * The kerb face under one side of a pavement, clipped the same way the
 * pavement above it is, so the two never disagree about where they stop.
 */
function pavedKerb(
  builder: MeshBuilder,
  line: Polyline,
  lateral: number,
  roadY: number,
  roads: RoadSurfaces,
): void {
  // Tested a hair outboard, so the kerb survives sitting exactly on its own
  // road edge but still vanishes where a crossing carriageway takes over.
  const probe = lateral >= 0 ? 0.2 : -0.2;
  for (const seg of segmentsOf(line)) {
    const nx = -seg.ty;
    const ny = seg.tx;
    const at = (d: number, lat: number): Vec2 => [
      seg.ox + seg.tx * d + nx * lat,
      seg.oy + seg.ty * d + ny * lat,
    ];
    const normal: P3 = [-seg.ty, 0, -seg.tx];
    const runs = keptRuns(
      seg.len,
      PAVE_STEP_M,
      (f, t) =>
        !roads.coversQuad([at(f, lateral), at(t, lateral), at(t, lateral + probe), at(f, lateral + probe)]),
    );
    for (const [a, b] of runs) {
      const p = at(a, lateral);
      const q = at(b, lateral);
      builder.quad(
        worldToThree(p[0], p[1], roadY),
        worldToThree(q[0], q[1], roadY),
        worldToThree(q[0], q[1], Y.sidewalk),
        worldToThree(p[0], p[1], Y.sidewalk),
        normal,
        C.kerb,
      );
    }
  }
}

/**
 * The pavement that wraps a junction, as a ring of cells around the crossing
 * point with everything on tarmac removed.
 *
 * This replaces four axis-aligned boxes, which were the right shape only where
 * two streets crossed square to the world axes — i.e. on the synthetic grid
 * and almost nowhere on a real one. A clipped ring needs to know nothing about
 * the crossing angle, the number of legs, or how wide either street is, and
 * the four corner fillets fall out of it on their own.
 */
function junctionApron(
  builder: MeshBuilder,
  at: Vec2,
  r0: number,
  r1: number,
  height: number,
  color: THREE.Color,
  roads: RoadSurfaces,
): void {
  const rings = Math.max(1, Math.ceil((r1 - r0) / 1.2));
  for (let ri = 0; ri < rings; ri++) {
    const ra = r0 + ((r1 - r0) * ri) / rings;
    const rb = r0 + ((r1 - r0) * (ri + 1)) / rings;
    for (let a = 0; a < APRON_SEGMENTS; a++) {
      const t0 = (a / APRON_SEGMENTS) * Math.PI * 2;
      const t1 = ((a + 1) / APRON_SEGMENTS) * Math.PI * 2;
      const corners: [Vec2, Vec2, Vec2, Vec2] = [
        [at[0] + Math.cos(t0) * ra, at[1] + Math.sin(t0) * ra],
        [at[0] + Math.cos(t1) * ra, at[1] + Math.sin(t1) * ra],
        [at[0] + Math.cos(t1) * rb, at[1] + Math.sin(t1) * rb],
        [at[0] + Math.cos(t0) * rb, at[1] + Math.sin(t0) * rb],
      ];
      if (roads.coversQuad(corners)) continue;
      builder.flatQuad(corners, height, color);
    }
  }
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
  const roadSurfaces = new RoadSurfaces(scene.roads, lines);

  scene.roads.forEach((road, i) => {
    const line = lines[i];
    const half = carriagewayHalfWidth(road);
    // A hair of vertical separation per road guarantees a stable draw order
    // where carriageways overlap at intersections.
    const yRoad = Y.road + i * 0.0009;
    const col = road.road_class === 'arterial' ? C.asphaltArterial : C.asphalt;

    stripe(surface, line, { from: 0, to: line.length }, 0, half * 2, yRoad, col);

    if (!road.has_sidewalk) return;
    // No arc-length holes any more: where a pavement has to stop is decided by
    // the carriageways it actually meets, not by an interval computed from one
    // crossing road's half-width as if the two streets met at a right angle.
    for (const side of [1, -1]) {
      pavedStrip(
        paving,
        line,
        side * (half + 0.02),
        side * (half + SIDEWALK_W),
        Y.sidewalk,
        C.sidewalk,
        roadSurfaces,
      );
      // Kerb face, so the pavement reads as raised rather than painted.
      pavedKerb(paving, line, side * (half + 0.02), yRoad, roadSurfaces);
    }
  });

  // Pavement wrapping each junction, so the strips either side of it join up.
  const aprons = new Map<string, { at: Vec2; r0: number; r1: number }>();
  scene.roads.forEach((road, i) => {
    if (!road.has_sidewalk) return;
    const hx = carriagewayHalfWidth(road);
    for (const c of crossings[i]) {
      const key = `${c.at[0].toFixed(2)}:${c.at[1].toFixed(2)}`;
      const r0 = Math.min(hx, c.half);
      const r1 = Math.max(hx, c.half) + SIDEWALK_W;
      const seen = aprons.get(key);
      if (seen) {
        seen.r0 = Math.min(seen.r0, r0);
        seen.r1 = Math.max(seen.r1, r1);
      } else {
        aprons.set(key, { at: c.at, r0, r1 });
      }
    }
  });
  for (const { at, r0, r1 } of aprons.values()) {
    junctionApron(paving, at, r0, r1, Y.sidewalk, C.sidewalk, roadSurfaces);
  }

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
    const h = carriagewayHalfWidth(road);
    const w = road.lane_width_m;
    const forward = road.lanes_forward;
    const backward = road.lanes_backward;
    // Right-hand traffic: a driver going the way the centreline runs keeps to
    // its RIGHT, which is negative lateral. So the forward lanes occupy
    // `[-h, -h + forward*w]` and the opposing ones the rest, and the division
    // between the two directions is where the forward lanes run out. On a
    // one-way street that lands on the far kerb, which is exactly right --
    // there is no opposing traffic, and its left edge is what sits there.
    const divide = -h + forward * w;
    const oncoming = backward > 0;

    const holes: Interval[] = crossings[i].map((c) => ({
      from: c.s - c.half - 1.5,
      to: c.s + c.half + 1.5,
    }));
    const spans = subtractIntervals(line.length, holes);

    const solid = (span: Interval, lateral: number, color: THREE.Color, width = LINE_W) =>
      stripe(markings, line, span, lateral, width, Y.marking, color);
    const broken = (span: Interval, lateral: number, color: THREE.Color) => {
      for (const run of dashRuns(span, DASH_M, GAP_M)) {
        stripe(markings, line, run, lateral, LINE_W, Y.marking, color);
      }
    };

    for (const span of spans) {
      // The line between opposing directions. Only drawn where there IS
      // opposing traffic -- a one-way street's `divide` is its far kerb.
      if (oncoming) {
        switch (road.center_marking) {
          case 'double_yellow':
            for (const o of [DOUBLE_GAP_M / 2, -DOUBLE_GAP_M / 2]) {
              solid(span, divide + o, C.markYellow);
            }
            break;
          case 'broken_yellow':
            broken(span, divide, C.markYellow);
            break;
          case 'solid_yellow':
            solid(span, divide, C.markYellow);
            break;
          case 'solid_white':
            solid(span, divide, C.markWhite);
            break;
          case 'dashed_white':
            broken(span, divide, C.markWhite);
            break;
          default:
            break;
        }
      }

      // Lane lines within one direction are broken white, both sides of the
      // divide. Laid out from the kerb the direction starts at, so a one-way
      // street's lines fall between its lanes instead of on its kerb.
      for (let k = 1; k < forward; k++) broken(span, -h + k * w, C.markWhite);
      for (let k = 1; k < backward; k++) broken(span, divide + k * w, C.markWhite);

      // Edge lines. A driver's own right-hand edge is white; the left edge of
      // a ONE-WAY roadway is yellow, which is the only place a yellow edge
      // line belongs. Not painted on residential streets or service alleys --
      // MUTCD reserves them for the higher-volume classes, and an alley with
      // a crisp painted edge on both sides does not read as an alley.
      if (road.road_class === 'arterial' || road.road_class === 'collector') {
        solid(span, -h + EDGE_INSET_M, C.markWhite);
        solid(span, h - EDGE_INSET_M, oncoming ? C.markWhite : C.markYellow);
      }

      // Stop bar across the approach lanes on the near side of a junction.
      for (const c of crossings[i]) {
        for (const [lanes, from, sign] of [
          [forward, -h, 1],
          [backward, divide, -1],
        ] as const) {
          if (lanes === 0) continue;
          const at = c.s - sign * (c.half + 2.2);
          if (at < span.from || at > span.to) continue;
          solid(
            { from: at - STOP_BAR_M / 2, to: at + STOP_BAR_M / 2 },
            from + (lanes * w) / 2,
            C.markWhite,
            lanes * w - 0.2,
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
  const octGeo = new THREE.CylinderGeometry(0.42, 0.42, PLATE_T, 8);
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
    // Three-space forward, for a world heading: world (cos h, sin h) -> (x, -z).
    const fx = Math.cos(s.heading);
    const fz = -Math.sin(s.heading);
    q.identity();
    // The post is bolted BEHIND the plate, not run through the middle of it.
    // At the same x/z its 0.05 m radius stuck 0.022 m out past the face, which
    // is 0.028 m proud of the plate — so a grey stripe was drawn straight down
    // the middle of every octagon.
    posts.setMatrixAt(
      i,
      m.compose(
        pos.set(x - fx * POST_SETBACK, 0, z - fz * POST_SETBACK),
        q,
        new THREE.Vector3(1, POST_H, 1),
      ),
    );
    q.setFromAxisAngle(up, s.heading);
    octs.setMatrixAt(i, m.compose(pos.set(x, SIGN_Y, z), q, one));
    faces.setMatrixAt(
      i,
      m.compose(pos.set(x + fx * FACE_PROUD, SIGN_Y, z + fz * FACE_PROUD), q, one),
    );
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

    // A blade's plane faces along its own +x once rotated, so this is the
    // direction it has to step to leave the post behind it.
    const bladeAngle = s.heading + Math.PI / 2;
    const bx = Math.cos(bladeAngle) * BLADE_PROUD;
    const bz = -Math.sin(bladeAngle) * BLADE_PROUD;

    if (s.kind === 'speed_limit') {
      const tex = speedLimitTexture(s.text);
      const mat = new THREE.MeshBasicNodeMaterial({
        map: tex,
        side: THREE.DoubleSide,
      });
      const geo = new THREE.PlaneGeometry(0.62, 0.93);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(x + bx, 2.0, z + bz);
      mesh.rotation.y = bladeAngle;
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
    // Stack blades down the post so two street names never overlap, and stand
    // them clear of it so the post does not draw over the text.
    mesh.position.set(x + bx, 3.0 - nth * 0.56, z + bz);
    // The blade runs parallel to its own carriageway, so the plane's normal is
    // perpendicular to the street's heading.
    mesh.rotation.y = bladeAngle;
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
