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

export class MeshBuilder {
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

  /** Flat horizontal quad at height `h`, given world-plane corners. */
  flatQuad(
    corners: [Vec2, Vec2, Vec2, Vec2],
    h: number,
    color: THREE.Color,
    uvScale: [number, number] = [1, 1],
  ): void {
    const [p0, p1, p2, p3] = corners.map((p) => worldToThree(p[0], p[1], h));
    this.quad(p0, p1, p2, p3, [0, 1, 0], color, uvScale);
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
  const a0 = line.offsetAt(span.from, lateral + width / 2);
  const a1 = line.offsetAt(span.from, lateral - width / 2);
  const b1 = line.offsetAt(span.to, lateral - width / 2);
  const b0 = line.offsetAt(span.to, lateral + width / 2);
  builder.flatQuad([a1, b1, b0, a0], height, color);
}
