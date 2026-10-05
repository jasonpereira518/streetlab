/**
 * Accumulates triangles into a single BufferGeometry.
 *
 * The whole static city is assembled from a handful of these, which is what
 * keeps the draw-call budget low: one builder per material, not one mesh per
 * kerbstone. Coordinates are already in three.js space — see `worldToThree`.
 */
import * as THREE from 'three/webgpu';
import type { Vec2 } from '../schema';

export type P3 = [number, number, number];

/**
 * World (x east, y north) -> three.js (x, height, -y). Declared once here so no
 * other module has to remember the sign flip.
 */
export const worldToThree = (x: number, y: number, h = 0): P3 => [x, h, -y];

/** World (x, y) -> ground height in metres. */
export type GroundFn = (x: number, y: number) => number;

/**
 * Longest edge a ground-following quad may have. The terrain is bilinear on a
 * 4 m grid; a flat quad longer than that bridges the curvature between samples
 * and either floats over a crest or buries itself in a dip.
 */
export const GROUND_STEP_M = 3;

export class MeshBuilder {
  /**
   * When set, every `flatQuad` and `wall` is draped over this ground: heights
   * passed in become offsets above it, and long quads are subdivided so they
   * follow it. Left null, geometry is exactly as before -- flat at the given
   * heights -- which is what a scene without terrain gets.
   */
  ground: GroundFn | null = null;

  private pos: number[] = [];
  private nrm: number[] = [];
  private uvs: number[] = [];
  private col: number[] = [];
  private useColor: boolean;

  constructor(useColor = true) {
    this.useColor = useColor;
  }

  get triangleCount(): number {
    return this.pos.length / 9;
  }

  get isEmpty(): boolean {
    return this.pos.length === 0;
  }

  tri(a: P3, b: P3, c: P3, normal: P3, color: THREE.Color, uv: [number, number][] = [[0, 0], [1, 0], [1, 1]]): void {
    for (const [i, v] of [a, b, c].entries()) {
      this.pos.push(v[0], v[1], v[2]);
      this.nrm.push(normal[0], normal[1], normal[2]);
      this.uvs.push(uv[i][0], uv[i][1]);
      if (this.useColor) this.col.push(color.r, color.g, color.b);
    }
  }

  /** Counter-clockwise quad `a b c d` as two triangles. */
  quad(
    a: P3,
    b: P3,
    c: P3,
    d: P3,
    normal: P3,
    color: THREE.Color,
    uvScale: [number, number] = [1, 1],
  ): void {
    const [u, v] = uvScale;
    this.tri(a, b, c, normal, color, [[0, 0], [u, 0], [u, v]]);
    this.tri(a, c, d, normal, color, [[0, 0], [u, v], [0, v]]);
  }

  /**
   * Horizontal quad at height `h`, given counter-clockwise world-plane corners.
   * With `ground` set, `h` is above the ground and the quad is draped over it.
   */
  flatQuad(
    corners: [Vec2, Vec2, Vec2, Vec2],
    h: number,
    color: THREE.Color,
    uvScale: [number, number] = [1, 1],
  ): void {
    const ground = this.ground;
    if (!ground) {
      const [p0, p1, p2, p3] = corners.map((p) => worldToThree(p[0], p[1], h));
      this.quad(p0, p1, p2, p3, [0, 1, 0], color, uvScale);
      return;
    }
    const [c0, c1, c2, c3] = corners;
    const edge = (a: Vec2, b: Vec2) => Math.hypot(b[0] - a[0], b[1] - a[1]);
    const nu = Math.max(1, Math.ceil(Math.max(edge(c0, c1), edge(c3, c2)) / GROUND_STEP_M));
    const nv = Math.max(1, Math.ceil(Math.max(edge(c0, c3), edge(c1, c2)) / GROUND_STEP_M));
    const at = (u: number, v: number): P3 => {
      const x = (1 - v) * ((1 - u) * c0[0] + u * c1[0]) + v * ((1 - u) * c3[0] + u * c2[0]);
      const y = (1 - v) * ((1 - u) * c0[1] + u * c1[1]) + v * ((1 - u) * c3[1] + u * c2[1]);
      return worldToThree(x, y, ground(x, y) + h);
    };
    const [su, sv] = uvScale;
    for (let i = 0; i < nu; i++) {
      for (let j = 0; j < nv; j++) {
        const a = at(i / nu, j / nv);
        const b = at((i + 1) / nu, j / nv);
        const c = at((i + 1) / nu, (j + 1) / nv);
        const d = at(i / nu, (j + 1) / nv);
        this.quad(a, b, c, d, upNormal(a, b, c), color, [su / nu, sv / nv]);
      }
    }
  }

  /**
   * Vertical face from world `p` to `q`, spanning heights `h0`..`h1`. With
   * `ground` set both are above the ground at each point along it.
   */
  wall(p: Vec2, q: Vec2, h0: number, h1: number, normal: P3, color: THREE.Color): void {
    const ground = this.ground;
    const n = ground
      ? Math.max(1, Math.ceil(Math.hypot(q[0] - p[0], q[1] - p[1]) / GROUND_STEP_M))
      : 1;
    for (let i = 0; i < n; i++) {
      const f0 = i / n;
      const f1 = (i + 1) / n;
      const x0 = p[0] + (q[0] - p[0]) * f0;
      const y0 = p[1] + (q[1] - p[1]) * f0;
      const x1 = p[0] + (q[0] - p[0]) * f1;
      const y1 = p[1] + (q[1] - p[1]) * f1;
      const g0 = ground ? ground(x0, y0) : 0;
      const g1 = ground ? ground(x1, y1) : 0;
      this.quad(
        worldToThree(x0, y0, g0 + h0),
        worldToThree(x1, y1, g1 + h0),
        worldToThree(x1, y1, g1 + h1),
        worldToThree(x0, y0, g0 + h1),
        normal,
        color,
      );
    }
  }

  /** Axis-aligned box in world coordinates, from `h0` to `h1`. */
  box(
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    h0: number,
    h1: number,
    color: THREE.Color,
  ): void {
    const c: Vec2[] = [
      [x0, y0],
      [x1, y0],
      [x1, y1],
      [x0, y1],
    ];
    // Top.
    this.flatQuad([c[0], c[1], c[2], c[3]], h1, color);
    // Sides.
    for (let i = 0; i < 4; i++) {
      const a = c[i];
      const b = c[(i + 1) % 4];
      const dx = b[0] - a[0];
      const dy = b[1] - a[1];
      const len = Math.hypot(dx, dy) || 1;
      // Outward normal for a CCW ring is the right-hand normal of the edge.
      const n: P3 = [dy / len, 0, dx / len];
      this.quad(
        worldToThree(a[0], a[1], h0),
        worldToThree(b[0], b[1], h0),
        worldToThree(b[0], b[1], h1),
        worldToThree(a[0], a[1], h1),
        n,
        color,
      );
    }
  }

  build(): THREE.BufferGeometry {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(this.pos, 3));
    g.setAttribute('normal', new THREE.Float32BufferAttribute(this.nrm, 3));
    g.setAttribute('uv', new THREE.Float32BufferAttribute(this.uvs, 2));
    if (this.useColor) {
      g.setAttribute('color', new THREE.Float32BufferAttribute(this.col, 3));
    }
    g.computeBoundingSphere();
    return g;
  }
}

/** Unit normal of triangle `a b c`, flipped if need be to point up. */
function upNormal(a: P3, b: P3, c: P3): P3 {
  const ux = b[0] - a[0];
  const uy = b[1] - a[1];
  const uz = b[2] - a[2];
  const vx = c[0] - a[0];
  const vy = c[1] - a[1];
  const vz = c[2] - a[2];
  let nx = uy * vz - uz * vy;
  let ny = uz * vx - ux * vz;
  let nz = ux * vy - uy * vx;
  const len = Math.hypot(nx, ny, nz);
  if (len < 1e-12) return [0, 1, 0];
  if (ny < 0) {
    nx = -nx;
    ny = -ny;
    nz = -nz;
  }
  return [nx / len, ny / len, nz / len];
}

/* ------------------------------------------------------------------ */
/* Polyline helpers                                                    */
/* ------------------------------------------------------------------ */

export interface PolylinePoint {
  x: number;
  y: number;
  /** Unit tangent. */
  tx: number;
  ty: number;
}

/** Arc-length lookup over an open polyline, used to lay out road furniture. */
export class Polyline {
  readonly points: Vec2[];
  readonly cum: number[];
  readonly length: number;

  constructor(points: Vec2[]) {
    this.points = points;
    this.cum = [0];
    let total = 0;
    for (let i = 1; i < points.length; i++) {
      total += Math.hypot(
        points[i][0] - points[i - 1][0],
        points[i][1] - points[i - 1][1],
      );
      this.cum.push(total);
    }
    this.length = total;
  }

  at(s: number): PolylinePoint {
    const t = Math.max(0, Math.min(this.length, s));
    let i = 1;
    while (i < this.cum.length - 1 && this.cum[i] < t) i++;
    const a = this.points[i - 1];
    const b = this.points[i];
    const segLen = this.cum[i] - this.cum[i - 1] || 1;
    const f = (t - this.cum[i - 1]) / segLen;
    const tx = (b[0] - a[0]) / segLen;
    const ty = (b[1] - a[1]) / segLen;
    return {
      x: a[0] + (b[0] - a[0]) * f,
      y: a[1] + (b[1] - a[1]) * f,
      tx,
      ty,
    };
  }

  /** Point offset `lateral` metres to the left of travel at arc length `s`. */
  offsetAt(s: number, lateral: number): Vec2 {
    const p = this.at(s);
    return [p.x - p.ty * lateral, p.y + p.tx * lateral];
  }
}

export interface Interval {
  from: number;
  to: number;
}

/**
 * Subtract `holes` from `[0, length]`, returning the surviving spans. Used to
 * stop lane markings, kerbs and sidewalks at intersections.
 */
export function subtractIntervals(length: number, holes: Interval[]): Interval[] {
  const sorted = [...holes]
    .map((h) => ({ from: Math.max(0, h.from), to: Math.min(length, h.to) }))
    .filter((h) => h.to > h.from)
    .sort((a, b) => a.from - b.from);

  const out: Interval[] = [];
  let cursor = 0;
  for (const h of sorted) {
    if (h.from > cursor) out.push({ from: cursor, to: h.from });
    cursor = Math.max(cursor, h.to);
  }
  if (cursor < length) out.push({ from: cursor, to: length });
  return out.filter((i) => i.to - i.from > 0.05);
}

/** Split a span into dash/gap runs, centred so dashes look regular. */
export function dashRuns(
  span: Interval,
  dash: number,
  gap: number,
): Interval[] {
  const out: Interval[] = [];
  const period = dash + gap;
  const n = Math.max(1, Math.round((span.to - span.from + gap) / period));
  const used = n * period - gap;
  let s = span.from + ((span.to - span.from) - used) / 2;
  for (let i = 0; i < n; i++) {
    const from = Math.max(span.from, s);
    const to = Math.min(span.to, s + dash);
    if (to > from) out.push({ from, to });
    s += period;
  }
  return out;
}

/**
 * Beyond this, a mitred corner would reach further than this many times the
 * lateral offset, so the join is bevelled instead. 4 allows turns up to ~151°.
 */
const MITER_LIMIT = 4;

/**
 * Lay a flat band between two lateral offsets along a polyline span, following
 * every vertex inside the span.
 *
 * Each interior vertex gets a mitred join, so neighbouring segments share
 * their corners exactly: no wedge gaps on the outside of a bend, no overlap on
 * the inside, and the tarmac, the paint on it and the pavement beside it all
 * agree on where the road is. (This used to be a single quad from the span's
 * first point to its last, which on a bent road is the chord, not the road.)
 */
export function ribbon(
  builder: MeshBuilder,
  line: Polyline,
  span: Interval,
  latA: number,
  latB: number,
  height: number,
  color: THREE.Color,
): void {
  const from = Math.max(0, span.from);
  const to = Math.min(line.length, span.to);
  if (to - from < 1e-6) return;
  const lo = Math.min(latA, latB);
  const hi = Math.max(latA, latB);
  const { points: pts, cum } = line;

  const normal = (k: number): Vec2 => {
    const len = cum[k] - cum[k - 1];
    return [-(pts[k][1] - pts[k - 1][1]) / len, (pts[k][0] - pts[k - 1][0]) / len];
  };
  // Segments of (effectively) zero length have no direction; skip past them.
  const live = (k: number) => cum[k] - cum[k - 1] > 1e-6;

  let k = 1;
  while (k < pts.length - 1 && (cum[k] <= from || !live(k))) k++;
  const start = line.at(from);
  let prevLo: Vec2 = [start.x - start.ty * lo, start.y + start.tx * lo];
  let prevHi: Vec2 = [start.x - start.ty * hi, start.y + start.tx * hi];

  const emit = (nLo: Vec2, nHi: Vec2) => {
    builder.flatQuad([prevLo, nLo, nHi, prevHi], height, color);
    prevLo = nLo;
    prevHi = nHi;
  };

  for (; k < pts.length - 1 && cum[k] < to; k++) {
    if (!live(k)) continue;
    let j = k + 1;
    while (j < pts.length && !live(j)) j++;
    if (j >= pts.length) break;
    const v = pts[k];
    const nIn = normal(k);
    const nOut = normal(j);
    const mx = nIn[0] + nOut[0];
    const my = nIn[1] + nOut[1];
    const mLen = Math.hypot(mx, my);
    const cosHalf = mLen / 2;
    if (mLen > 1e-9 && 1 / cosHalf <= MITER_LIMIT) {
      const scale = 1 / (mLen * cosHalf);
      const m: Vec2 = [mx * scale, my * scale];
      emit([v[0] + m[0] * lo, v[1] + m[1] * lo], [v[0] + m[0] * hi, v[1] + m[1] * hi]);
    } else {
      // Hairpin: finish square on the incoming segment, fill the bevel, and
      // restart square on the outgoing one.
      emit([v[0] + nIn[0] * lo, v[1] + nIn[1] * lo], [v[0] + nIn[0] * hi, v[1] + nIn[1] * hi]);
      emit([v[0] + nOut[0] * lo, v[1] + nOut[1] * lo], [v[0] + nOut[0] * hi, v[1] + nOut[1] * hi]);
    }
    k = j - 1;
  }

  const end = line.at(to);
  emit([end.x - end.ty * lo, end.y + end.tx * lo], [end.x - end.ty * hi, end.y + end.tx * hi]);
}

/** Lay a constant-width strip of colour along a polyline span. */
export function stripe(
  builder: MeshBuilder,
  line: Polyline,
  span: Interval,
  lateral: number,
  width: number,
  height: number,
  color: THREE.Color,
): void {
  ribbon(builder, line, span, lateral - width / 2, lateral + width / 2, height, color);
}
