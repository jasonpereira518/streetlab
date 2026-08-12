/**
 * Arc-length parameterised closed routes, used by the mock simulator to drive
 * the ego vehicle and the traffic agents. Corners are filleted with circular
 * arcs so headings stay continuous through turns — a chase camera following a
 * polyline with hard corners looks broken.
 */
import { angleDelta } from '../units';
import type { Vec2 } from '../schema';

export interface RouteSample {
  x: number;
  y: number;
  /** radians, 0 = +x, CCW positive */
  heading: number;
  /** 1/m, positive = turning left */
  curvature: number;
}

type Element =
  | {
      kind: 'line';
      x0: number;
      y0: number;
      dx: number;
      dy: number;
      heading: number;
      len: number;
      s0: number;
    }
  | {
      kind: 'arc';
      cx: number;
      cy: number;
      r: number;
      a0: number;
      /** signed sweep, positive = left turn */
      sweep: number;
      len: number;
      s0: number;
    };

export class Route {
  readonly length: number;
  private readonly elements: Element[];

  constructor(elements: Element[], length: number) {
    this.elements = elements;
    this.length = length;
  }

  /** Wrap a distance into `[0, length)`. */
  wrap(s: number): number {
    const m = s % this.length;
    return m < 0 ? m + this.length : m;
  }

  /** Signed gap from `a` to `b` along the loop, in `(-length/2, length/2]`. */
  gap(a: number, b: number): number {
    let d = this.wrap(b - a);
    if (d > this.length / 2) d -= this.length;
    return d;
  }

  /**
   * Pose at arc length `s`, optionally shifted `lateral` metres to the left of
   * the centreline (negative shifts right, i.e. toward the kerb).
   */
  sample(s: number, lateral = 0): RouteSample {
    const t = this.wrap(s);
    const el = this.elementAt(t);
    const local = t - el.s0;

    let x: number;
    let y: number;
    let heading: number;
    let curvature: number;

    if (el.kind === 'line') {
      x = el.x0 + el.dx * local;
      y = el.y0 + el.dy * local;
      heading = el.heading;
      curvature = 0;
    } else {
      const dir = Math.sign(el.sweep);
      const a = el.a0 + (local / el.r) * dir;
      x = el.cx + el.r * Math.cos(a);
      y = el.cy + el.r * Math.sin(a);
      heading = a + (dir > 0 ? Math.PI / 2 : -Math.PI / 2);
      curvature = dir / el.r;
    }

    if (lateral !== 0) {
      // Left normal of the heading.
      x += -Math.sin(heading) * lateral;
      y += Math.cos(heading) * lateral;
    }
    return { x, y, heading, curvature };
  }

  /** Densely sampled polyline, for plan ribbons and minimap previews. */
  polyline(s0: number, len: number, step: number, lateral = 0): Vec2[] {
    const out: Vec2[] = [];
    const n = Math.max(2, Math.ceil(len / step));
    for (let i = 0; i <= n; i++) {
      const p = this.sample(s0 + (len * i) / n, lateral);
      out.push([p.x, p.y]);
    }
    return out;
  }

  private elementAt(s: number): Element {
    // Elements are short and few (8 for a rectangle loop); a linear scan from a
    // binary-search seed is simpler than caching and never mispredicts.
    let lo = 0;
    let hi = this.elements.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.elements[mid].s0 <= s) lo = mid;
      else hi = mid - 1;
    }
    return this.elements[lo];
  }
}

/**
 * Build a closed route through `corners`, filleting every corner with an arc of
 * `radius`. Corners must describe a simple polygon with turns sharper than a
 * straight line and gentler than a full reversal.
 */
export function makeLoopRoute(corners: Vec2[], radius: number): Route {
  const n = corners.length;
  if (n < 3) throw new Error('makeLoopRoute needs at least 3 corners');

  // Per corner: incoming/outgoing unit direction, fillet tangent points.
  const dirs: Array<{ dx: number; dy: number; heading: number }> = [];
  for (let i = 0; i < n; i++) {
    const a = corners[i];
    const b = corners[(i + 1) % n];
    const dx = b[0] - a[0];
    const dy = b[1] - a[1];
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) throw new Error('makeLoopRoute: duplicate corner');
    dirs.push({ dx: dx / len, dy: dy / len, heading: Math.atan2(dy, dx) });
  }

  type Fillet = {
    inX: number;
    inY: number;
    outX: number;
    outY: number;
    cx: number;
    cy: number;
    a0: number;
    sweep: number;
    r: number;
  };
  const fillets: Fillet[] = [];

  for (let i = 0; i < n; i++) {
    const p = corners[i];
    const dIn = dirs[(i - 1 + n) % n]; // direction arriving at p
    const dOut = dirs[i]; // direction leaving p
    const turn = angleDelta(dOut.heading, dIn.heading);
    if (Math.abs(turn) < 1e-4) {
      fillets.push({
        inX: p[0],
        inY: p[1],
        outX: p[0],
        outY: p[1],
        cx: p[0],
        cy: p[1],
        a0: 0,
        sweep: 0,
        r: radius,
      });
      continue;
    }
    const tangent = radius * Math.tan(Math.abs(turn) / 2);
    const inX = p[0] - dIn.dx * tangent;
    const inY = p[1] - dIn.dy * tangent;
    const outX = p[0] + dOut.dx * tangent;
    const outY = p[1] + dOut.dy * tangent;
    // Centre sits on the normal at the entry tangent point, on the inside of
    // the turn: left normal for a left turn, right normal for a right turn.
    const sign = turn > 0 ? 1 : -1;
    const cx = inX + sign * -dIn.dy * radius;
    const cy = inY + sign * dIn.dx * radius;
    const a0 = Math.atan2(inY - cy, inX - cx);
    fillets.push({ inX, inY, outX, outY, cx, cy, a0, sweep: turn, r: radius });
  }

  const elements: Element[] = [];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const f = fillets[i];
    // Arc through corner i.
    if (f.sweep !== 0) {
      const len = Math.abs(f.sweep) * f.r;
      elements.push({
        kind: 'arc',
        cx: f.cx,
        cy: f.cy,
        r: f.r,
        a0: f.a0,
        sweep: f.sweep,
        len,
        s0: s,
      });
      s += len;
    }
    // Straight from corner i's exit to corner i+1's entry.
    const next = fillets[(i + 1) % n];
    const dx = next.inX - f.outX;
    const dy = next.inY - f.outY;
    const len = Math.hypot(dx, dy);
    if (len < 1e-6) {
      throw new Error('makeLoopRoute: fillet radius too large for a segment');
    }
    elements.push({
      kind: 'line',
      x0: f.outX,
      y0: f.outY,
      dx: dx / len,
      dy: dy / len,
      heading: Math.atan2(dy, dx),
      len,
      s0: s,
    });
    s += len;
  }

  return new Route(elements, s);
}

/** Axis-aligned rectangle loop, traversed in the order the corners are given. */
export function makeRectRoute(
  minX: number,
  minY: number,
  maxX: number,
  maxY: number,
  radius: number,
  clockwise: boolean,
): Route {
  const ccw: Vec2[] = [
    [minX, minY],
    [maxX, minY],
    [maxX, maxY],
    [minX, maxY],
  ];
  return makeLoopRoute(clockwise ? [...ccw].reverse() : ccw, radius);
}
