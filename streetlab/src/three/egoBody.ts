/**
 * The ego's exterior: a low two-door fastback in satin champagne, with a glass
 * canopy, flat aero wheel covers and full-width light bars.
 *
 * Built from side profiles rather than boxes. A profile (X forward, Y up) is
 * extruded across the car's width with a large bevel for round shoulders, then
 * tessellated and bent by one deformation (`deform`) that pulls the roof in
 * (tumblehome), pinches the nose and tail in plan, and bows the nose so its
 * corners sweep back. Everything that sits on the skin -- windows, seams, lamp
 * bars -- is placed in the undeformed frame and bent by the same function, so
 * it stays on the surface.
 *
 * Frame: as ego.ts. Origin at the centre of the footprint on the ground, +X
 * forward, +Y up, -Z is the car's left.
 *
 * Proportions are authored for a 4.7 x 1.45 m reference car and scaled to the
 * `Size` the sim reports, so the drawn car is the car that collides.
 */
import * as THREE from 'three/webgpu';
import { attribute } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { TessellateModifier } from 'three/addons/modifiers/TessellateModifier.js';
import type { Size } from '../schema';
import { paint } from './ego';

export const EGO_PAINT = {
  body: '#DCCDAE',
  glass: '#161B21',
  trim: '#141518',
  seam: '#5C5040',
  cover: '#CFC0A0',
  coverMark: '#B3A27F',
  tyre: '#1B1C1E',
} as const;

/**
 * How each paint reflects: [roughness, metalness]. Satin metallic body and
 * covers, glossy glass, satin black plastics, matte rubber.
 */
const FINISH: Record<keyof typeof EGO_PAINT, [number, number]> = {
  body: [0.36, 0.7],
  glass: [0.04, 0],
  trim: [0.42, 0],
  seam: [0.5, 0.2],
  cover: [0.36, 0.7],
  coverMark: [0.38, 0.55],
  tyre: [0.88, 0],
};

/**
 * Tag every vertex with the finish of the paint it was coloured with. Done
 * once, on the merged buffer, by nearest palette colour.
 */
function addFinish(geo: THREE.BufferGeometry): THREE.BufferGeometry {
  const keys = Object.keys(EGO_PAINT) as Array<keyof typeof EGO_PAINT>;
  const pal = keys.map((k) => new THREE.Color(EGO_PAINT[k]).convertSRGBToLinear());
  const col = geo.attributes.color as THREE.BufferAttribute;
  const out = new Float32Array(col.count * 2);
  for (let i = 0; i < col.count; i++) {
    const r = col.getX(i);
    const g = col.getY(i);
    const b = col.getZ(i);
    let best = 0;
    let bestD = Infinity;
    for (let k = 0; k < pal.length; k++) {
      const dd = (pal[k].r - r) ** 2 + (pal[k].g - g) ** 2 + (pal[k].b - b) ** 2;
      if (dd < bestD) [best, bestD] = [k, dd];
    }
    out.set(FINISH[keys[best]], i * 2);
  }
  geo.setAttribute('finish', new THREE.BufferAttribute(out, 2));
  return geo;
}

/** Unlit lens colours; ego.ts makes them glow. */
export const LENS = { head: '#DCE6EE', tail: '#5E1B1D', amber: '#6B4A1E' } as const;

const REF_L = 4.7;
const REF_H = 1.45;
/** Shoulder rounding of the main body, metres. */
const BEVEL = 0.085;
/** Plain-skin tessellation target, metres: fine enough for the bend to read. */
const TESS = 0.12;

/** Everything about the body other code needs to place things on it. */
export interface EgoBody {
  /** Body, glass and trim, one vertex-coloured buffer. */
  shell: THREE.BufferGeometry;
  /** Light bars, one buffer per independently lit group. */
  lamps: {
    head: THREE.BufferGeometry;
    tail: THREE.BufferGeometry;
    indicatorLeft: THREE.BufferGeometry;
    indicatorRight: THREE.BufferGeometry;
  };
  wheel: { radius: number; width: number; axleX: number; centreZ: number };
  /** Where the headlight beams start, right side (mirror z for the left). */
  beam: THREE.Vector3;
}

interface Dims {
  H: number;
  hx: number;
  hw: number;
  belt: number;
  /** Underside of the painted body. */
  sill: number;
  axleX: number;
  wheelR: number;
  archR: number;
}

/** How far the door panels sit inside the flares, as a fraction of half-width. */
const PANEL_INSET = 0.07;

const sm = (e: number) => {
  const t = Math.min(1, Math.max(0, e));
  return t * t * (3 - 2 * t);
};

/**
 * The one bend applied to every part of the skin. Below the beltline it gives
 * the sides their form -- flares over the wheels, door panels set in between,
 * a section that tucks under toward the sill -- and above it the tumblehome.
 * The flares peak at exactly the half-width, so the footprint is the sim's.
 */
function deform(p: THREE.Vector3, d: Dims): THREE.Vector3 {
  const u = p.x / d.hx;
  const t = Math.min(1, Math.max(0, (p.y - d.belt) / (d.H - d.belt)));
  const v = Math.min(1, Math.max(0, (p.y - d.sill) / (d.belt - d.sill)));

  // Flares: swell out of each arch lip. The front one fades ~0.35 m away; the
  // rear haunch is broader, runs back to the tail and up to the shoulder.
  const front = sm((d.archR + 0.4 - Math.hypot(p.x - d.axleX, p.y - d.wheelR)) / 0.32);
  // An ellipse, stretched along the car. Every term here is smooth: a kink in
  // this function is a kink in the reflections.
  const rear = sm((d.archR + 0.55 - Math.hypot((p.x + d.axleX) * 0.62, p.y - d.wheelR)) / 0.45);
  const a = front * (1 - sm((v - 0.75) / 0.25));
  const b = rear * (1 - sm((v - 0.8) / 0.2)); // done by the belt, where v stops
  const flare = a + b - a * b; // smooth union of the two
  let s = 1 - PANEL_INSET * (1 - flare);
  // Section: tucked in at the sill, fullest a third of the way up, easing in
  // again toward the shoulder.
  s *= 1 - 0.08 * (1 - sm(v / 0.45)) ** 2 - 0.02 * sm((v - 0.45) / 0.55);
  // A shallow scoop along the lower doors, between the arches.
  const between = 1 - sm((Math.abs(p.x) - (d.axleX - d.archR - 0.1)) / 0.15);
  s *= 1 - 0.018 * between * Math.exp(-(((v - 0.3) / 0.1) ** 2));

  // Tumblehome: roof ~70% of the shoulders, easing in from the belt -- a
  // power curve starts almost vertically there and creases the reflections.
  s *= 1 - 0.3 * t * t * (2 - t);
  s *= 1 - 0.07 * t * sm((-u - 0.1) / 0.8); // cabin narrows toward the tail
  s *= 1 - 0.14 * sm((u - 0.75) / 0.25); // nose pinch, clear of the front arch
  s *= 1 - 0.05 * sm((-u - 0.78) / 0.22); // tail pinch
  const z = p.z * s;
  const zz = Math.min(1, (z / d.hw) ** 2);
  const x = p.x - 0.2 * sm((u - 0.78) / 0.22) * zz + 0.04 * sm((-u - 0.85) / 0.15) * zz;
  // Crown across the width: the bonnet bulges, the roof a little less.
  const crown = (0.012 + 0.045 * sm((u - 0.15) / 0.4)) * (1 - zz) * sm((v - 0.55) / 0.45);
  return p.set(x, p.y + crown, z);
}

/**
 * Bend a geometry, carrying its normals through the bend exactly: each normal
 * is mapped by the inverse transpose of `deform`'s Jacobian (by finite
 * differences) instead of being re-averaged from the triangles. Metallic
 * paint shows every error in a normal, and averaged normals follow the
 * triangulation -- irregular slivers on the flat sides -- not the surface.
 */
function deformGeometry(geo: THREE.BufferGeometry, d: Dims): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const nor = geo.attributes.normal as THREE.BufferAttribute | undefined;
  const v = new THREE.Vector3();
  const p0 = new THREE.Vector3();
  const ex = new THREE.Vector3();
  const ey = new THREE.Vector3();
  const ez = new THREE.Vector3();
  const n = new THREE.Vector3();
  const J = new THREE.Matrix3();
  const e = 1e-3;
  for (let i = 0; i < pos.count; i++) {
    v.fromBufferAttribute(pos, i);
    if (nor) {
      deform(p0.copy(v), d);
      deform(ex.set(v.x + e, v.y, v.z), d).sub(p0).divideScalar(e);
      deform(ey.set(v.x, v.y + e, v.z), d).sub(p0).divideScalar(e);
      deform(ez.set(v.x, v.y, v.z + e), d).sub(p0).divideScalar(e);
      J.set(ex.x, ey.x, ez.x, ex.y, ey.y, ez.y, ex.z, ey.z, ez.z).invert().transpose();
      n.fromBufferAttribute(nor, i).applyMatrix3(J).normalize();
      nor.setXYZ(i, n.x, n.y, n.z);
      pos.setXYZ(i, p0.x, p0.y, p0.z);
    } else {
      deform(v, d);
      pos.setXYZ(i, v.x, v.y, v.z);
    }
  }
  pos.needsUpdate = true;
  if (nor) nor.needsUpdate = true;
}

function tessellate(geo: THREE.BufferGeometry, edge = TESS): THREE.BufferGeometry {
  const flat = geo.index ? geo.toNonIndexed() : geo;
  // Each pass splits only a triangle's longest edge, so a 4.7 m side needs
  // far more than log2(4.7 / edge) passes. Stopping early leaves T-junctions,
  // which crack open and speckle the paint.
  const out = new TessellateModifier(edge, 40).modify(flat);
  if (flat !== geo) flat.dispose();
  geo.dispose();
  flattenSides(out);
  return out;
}

/**
 * The flat sides' normals are exactly +-Z. The creased normals at their rim
 * are averaged with the bevel and lean outward, and the sides are triangulated
 * as long slivers reaching that rim, so tessellation smeared the lean across
 * the whole panel -- crumpled foil once the paint is metallic.
 */
function flattenSides(geo: THREE.BufferGeometry): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const nor = geo.attributes.normal as THREE.BufferAttribute | undefined;
  if (!nor) return;
  const tri = new THREE.Triangle();
  const n = new THREE.Vector3();
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 3) {
    tri.set(a.fromBufferAttribute(pos, i), b.fromBufferAttribute(pos, i + 1), c.fromBufferAttribute(pos, i + 2));
    tri.getNormal(n);
    if (Math.abs(n.z) < 0.9999) continue;
    const z = Math.sign(n.z);
    for (let k = 0; k < 3; k++) nor.setXYZ(i + k, 0, 0, z);
  }
}

/** Extrude a side profile across `width`, centred on z = 0. */
function extrudeProfile(shape: THREE.Shape, width: number, bevel: number): THREE.BufferGeometry {
  const geo = new THREE.ExtrudeGeometry(shape, {
    depth: width - bevel * 2,
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel,
    // Pull the bevel inside the outline, so the profile is the silhouette.
    bevelOffset: -bevel,
    bevelSegments: 8,
    curveSegments: 32,
  });
  geo.translate(0, 0, -(width - bevel * 2) / 2);
  // Normals now, on the clean extrusion: its bevels are regular strips of
  // quads. After tessellation they would be averaged over irregular slivers,
  // and that noise ripples every reflection. Tessellation interpolates these.
  return creasedNormals(geo, 0.6);
}

/**
 * Smooth normals across the curves, crisp past `crease` (rad): the tail edge
 * and the arches. Unlike three's toCreasedNormals, faces are weighted by
 * their corner angle and the flat sides are left out: the sides' long
 * sliver triangles fan into the shoulder's edge in uneven numbers, and an
 * even-handed average let them tug some of its normals sideways and not
 * others -- a ripple along every shoulder.
 */
function creasedNormals(geo: THREE.BufferGeometry, crease: number): THREE.BufferGeometry {
  const g = geo.index ? geo.toNonIndexed() : geo;
  if (g !== geo) geo.dispose();
  const pos = g.attributes.position as THREE.BufferAttribute;
  const faceN: THREE.Vector3[] = [];
  const angle: number[] = [];
  const key = (i: number) =>
    `${Math.round(pos.getX(i) * 1e4)},${Math.round(pos.getY(i) * 1e4)},${Math.round(pos.getZ(i) * 1e4)}`;
  const byPos = new Map<string, number[]>(); // position -> corner indices
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  for (let i = 0; i < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    const n = new THREE.Vector3().crossVectors(e1.subVectors(b, a), e2.subVectors(c, a)).normalize();
    const corners = [
      [a, b, c],
      [b, c, a],
      [c, a, b],
    ];
    for (let k = 0; k < 3; k++) {
      const [p, q, r] = corners[k];
      faceN[i + k] = n;
      angle[i + k] = e1.subVectors(q, p).angleTo(e2.subVectors(r, p)) || 0;
      const id = key(i + k);
      const list = byPos.get(id);
      if (list) list.push(i + k);
      else byPos.set(id, [i + k]);
    }
  }
  const cos = Math.cos(crease);
  const out = new Float32Array(pos.count * 3);
  const sum = new THREE.Vector3();
  for (const list of byPos.values()) {
    for (const i of list) {
      const own = faceN[i];
      sum.set(0, 0, 0);
      for (const j of list) {
        const n = faceN[j];
        const side = Math.abs(n.z) > 0.9999;
        // A flat side keeps its own normal; nothing else borrows from it.
        if (side && j !== i && !(Math.abs(own.z) > 0.9999)) continue;
        if (n.dot(own) < cos) continue;
        sum.addScaledVector(n, angle[j]);
      }
      if (sum.lengthSq() === 0) sum.copy(own);
      sum.normalize().toArray(out, i * 3);
    }
  }
  g.setAttribute('normal', new THREE.BufferAttribute(out, 3));
  return g;
}

/** A flat skin plate of `shape`, laid just proud of the side at z = +-hw. */
function sidePlate(shape: THREE.Shape, d: Dims, side: 1 | -1, proud = 0.004): THREE.BufferGeometry {
  const t = 0.004;
  const geo = new THREE.ExtrudeGeometry(shape, { depth: t, bevelEnabled: false, curveSegments: 8 });
  geo.translate(0, 0, side > 0 ? d.hw + proud - t : -d.hw - proud);
  return geo;
}

/** Colour each triangle by where it sits, in the undeformed frame. */
function colourTriangles(
  geo: THREE.BufferGeometry,
  pick: (c: THREE.Vector3, n: THREE.Vector3) => string,
): void {
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const colors = new Float32Array(pos.count * 3);
  const a = new THREE.Vector3();
  const b = new THREE.Vector3();
  const c = new THREE.Vector3();
  const tri = new THREE.Triangle();
  const n = new THREE.Vector3();
  const centre = new THREE.Vector3();
  const cache = new Map<string, THREE.Color>();
  for (let i = 0; i < pos.count; i += 3) {
    a.fromBufferAttribute(pos, i);
    b.fromBufferAttribute(pos, i + 1);
    c.fromBufferAttribute(pos, i + 2);
    tri.set(a, b, c);
    tri.getNormal(n);
    tri.getMidpoint(centre);
    const hex = pick(centre, n);
    let col = cache.get(hex);
    if (!col) cache.set(hex, (col = new THREE.Color(hex).convertSRGBToLinear()));
    for (let k = 0; k < 3; k++) col.toArray(colors, (i + k) * 3);
  }
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

/** Where the outline crosses height `y`: its frontmost or rearmost x. */
function outlineX(shape: THREE.Shape, y: number, front: boolean): number {
  const pts = shape.getPoints(24);
  let best = front ? -Infinity : Infinity;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    if ((p.y - y) * (q.y - y) > 0 || p.y === q.y) continue;
    const x = p.x + ((y - p.y) / (q.y - p.y)) * (q.x - p.x);
    best = front ? Math.max(best, x) : Math.min(best, x);
  }
  return best;
}

/** Give every part the same attribute set so they merge. */
function stripForMerge(g: THREE.BufferGeometry): THREE.BufferGeometry {
  for (const name of Object.keys(g.attributes)) {
    if (!['position', 'normal', 'color'].includes(name)) g.deleteAttribute(name);
  }
  return g;
}

const merge = (parts: THREE.BufferGeometry[]): THREE.BufferGeometry => {
  const g = mergeGeometries(parts.map(stripForMerge), false);
  for (const p of parts) p.dispose();
  if (!g) throw new Error('failed to merge ego geometry');
  g.computeBoundingSphere();
  return g;
};

/** A thin tube through skin points given in the undeformed frame. */
function lampBar(pts: THREE.Vector3[], d: Dims, radius: number, hex: string): THREE.BufferGeometry {
  const curve = new THREE.CatmullRomCurve3(pts.map((p) => deform(p.clone(), d)));
  return paint(new THREE.TubeGeometry(curve, pts.length * 8, radius, 10, false), hex);
}

/** A plate on the side skin, bent onto it, with its own normals. */
function skinPlate(shape: THREE.Shape, d: Dims, side: 1 | -1, hex: string, proud?: number) {
  const g = tessellate(paint(sidePlate(shape, d, side, proud), hex), 0.08);
  g.computeVertexNormals(); // flat plate: exactly +-Z, then bent with the skin
  deformGeometry(g, d);
  return g;
}

export function buildEgoBody(size: Size): EgoBody {
  const { length: L, width: W, height: H } = size;
  const kx = L / REF_L;
  const ky = H / REF_H;
  const X = (x: number) => x * kx;
  const Y = (y: number) => y * ky;

  // Big wheels in tight arches. The body over the front arch must stay thicker
  // than two bevels, or the inset side face folds over itself.
  const wheelR = 0.36 * ky;
  const archR = wheelR + 0.028;
  // Same wheelbase-to-length ratio as the sim's bicycle model (2.9 m on 4.7 m).
  const axleX = 0.3085 * L;
  const yS = Y(0.26); // underside of the painted body; the black skirt shows below
  const d: Dims = { H, hx: L / 2, hw: W / 2, belt: 0.95 * ky, sill: yS, axleX, wheelR, archR };
  /** Half-chord and end angle of an arch cut into a bottom edge at `y0`. */
  const archAt = (y0: number) => {
    const a = Math.asin((y0 - wheelR) / archR);
    return { a, dx: Math.cos(a) * archR };
  };

  const screenCurve = new THREE.CubicBezierCurve(
    new THREE.Vector2(X(1.18), Y(0.95)),
    new THREE.Vector2(X(0.85), Y(1.1)),
    new THREE.Vector2(X(0.45), Y(1.4)),
    new THREE.Vector2(X(0), Y(1.45)),
  );
  const roofCurve = new THREE.CubicBezierCurve(
    new THREE.Vector2(X(0), Y(1.45)),
    new THREE.Vector2(X(-0.6), Y(1.47)),
    new THREE.Vector2(X(-1.6), Y(1.22)),
    new THREE.Vector2(X(-2.29), Y(1.02)),
  );

  /* ---- painted upper body ---- */
  const body = new THREE.Shape();
  body.moveTo(X(2.29), Y(0.3));
  // Nose: short and round, the light bar across its crown.
  body.quadraticCurveTo(X(2.35), Y(0.36), X(2.35), Y(0.47));
  body.quadraticCurveTo(X(2.35), Y(0.63), X(2.22), Y(0.7));
  // Gently crowned bonnet up to the cowl, high over the front wheels.
  body.bezierCurveTo(X(1.95), Y(0.84), X(1.55), Y(0.92), X(1.18), Y(0.95));
  // Long, steeply raked windscreen into the roof...
  body.curves.push(screenCurve);
  // ...and a fastback down to a crisp, flat Kamm tail.
  body.curves.push(roofCurve);
  body.currentPoint.copy(roofCurve.v3);
  body.lineTo(X(-2.345), Y(0.97));
  body.lineTo(X(-2.35), Y(0.47));
  body.lineTo(X(-2.05), Y(0.45));
  body.quadraticCurveTo(X(-1.95), Y(0.45), X(-1.95), yS);
  // Bottom edge, rear to front, over both wheels.
  {
    const { a, dx } = archAt(yS);
    for (const cx of [-axleX, axleX]) {
      body.lineTo(cx - dx, yS);
      body.absarc(cx, wheelR, archR, Math.PI - a, a, true);
    }
  }
  body.lineTo(X(2.0), yS);
  body.quadraticCurveTo(X(2.2), yS, X(2.29), Y(0.3));

  // The canopy's edges sit exactly on outline vertices (the extrusion samples
  // each curve at t = k/32), so no wall triangle straddles them.
  const glassFrom = roofCurve.getPoint(20 / 32).x;
  const glassTo = screenCurve.getPoint(0).x;
  const shell = tessellate(extrudeProfile(body, W, BEVEL));
  colourTriangles(shell, (c, n) => {
    // Still undeformed: the flat sides are exactly +-Z.
    const side = Math.abs(n.z) > 0.999;
    // Wheel-arch liners: the faces inside the arch tunnels.
    for (const ax of [-axleX, axleX]) {
      // Only the tunnel walls (normal across the car's length), not the
      // rounded lip that faces outward.
      if (!side && Math.abs(n.z) < 0.02 && Math.hypot(c.x - ax, c.y - wheelR) < archR + 0.02) {
        return EGO_PAINT.trim;
      }
    }
    // Canopy: windscreen, roof and the shoulders over the side glass.
    if (!side && c.y > Y(0.6) && c.x > glassFrom && c.x < glassTo) return EGO_PAINT.glass;
    return EGO_PAINT.body;
  });

  /* ---- black skirt: rocker band, front lip, rear bumper ---- */
  // Three pieces, each stopping short of the arches: a skirt that ran under
  // the wheels would need its arch cut-outs to rise above its own top edge.
  const y0 = Y(0.17);
  const top = yS + 0.06;
  const clear = archR + 0.01; // half-chord of the arch at its widest
  const rear = new THREE.Shape();
  rear.moveTo(X(-2.345), Y(0.46));
  rear.lineTo(X(-1.98), Y(0.46));
  rear.lineTo(X(-1.98), top);
  rear.lineTo(-axleX - clear, top);
  rear.lineTo(-axleX - clear, y0);
  rear.lineTo(X(-2.2), y0);
  rear.quadraticCurveTo(X(-2.345), y0, X(-2.345), Y(0.3));
  const rocker = new THREE.Shape([
    new THREE.Vector2(-axleX + clear, y0),
    new THREE.Vector2(axleX - clear, y0),
    new THREE.Vector2(axleX - clear, top),
    new THREE.Vector2(-axleX + clear, top),
  ]);
  const lip = new THREE.Shape();
  lip.moveTo(axleX + clear, y0);
  lip.lineTo(X(2.2), y0);
  lip.quadraticCurveTo(X(2.34), y0, X(2.32), Y(0.31));
  lip.lineTo(X(2.1), top);
  lip.lineTo(axleX + clear, top);
  const skirtGeo = tessellate(
    merge([rear, rocker, lip].map((sh) => extrudeProfile(sh, W - 0.07, 0.035))),
  );
  colourTriangles(skirtGeo, () => EGO_PAINT.trim);

  /* ---- side glass, door seams, camera pods ---- */
  // The window runs under the canopy's shoulder, from the A-pillar back to a
  // swept point ahead of the body-colour sail.
  const under = BEVEL + 0.004; // stay on the flat side, under the shoulder
  const winTop = [
    ...screenCurve.getPoints(16).map((p) => new THREE.Vector2(p.x - 0.05, p.y - under)),
    ...roofCurve.getPoints(24).map((p) => new THREE.Vector2(p.x, p.y - under)),
  ].filter((p) => p.y > d.belt + 0.04 && p.x > X(-1.05));
  const win = new THREE.Shape();
  win.moveTo(winTop[0].x, d.belt + 0.02);
  for (const p of winTop) win.lineTo(p.x, p.y);
  win.lineTo(X(-1.34), d.belt + 0.05);
  win.lineTo(X(-1.0), d.belt + 0.02);

  const seam = (a: [number, number], b: [number, number]) => {
    const pa = new THREE.Vector2(X(a[0]), Y(a[1]));
    const pb = new THREE.Vector2(X(b[0]), Y(b[1]));
    const dir = pb.clone().sub(pa).normalize();
    const off = new THREE.Vector2(-dir.y, dir.x).multiplyScalar(0.005);
    return new THREE.Shape([
      pa.clone().add(off),
      pb.clone().add(off),
      pb.clone().sub(off),
      pa.clone().sub(off),
    ]);
  };
  // Camera pod: a small black wedge on the fender, ahead of the door.
  const pod = new THREE.Shape([
    new THREE.Vector2(X(1.0), Y(0.78)),
    new THREE.Vector2(X(1.12), Y(0.78)),
    new THREE.Vector2(X(1.0), Y(0.85)),
  ]);

  const skin: THREE.BufferGeometry[] = [];
  for (const side of [1, -1] as const) {
    skin.push(skinPlate(win, d, side, EGO_PAINT.glass));
    // Butterfly door: one long seam behind the front arch, one ahead of the sail.
    skin.push(skinPlate(seam([0.9, 0.39], [0.92, 0.88]), d, side, EGO_PAINT.seam, 0.003));
    skin.push(skinPlate(seam([-0.9, 0.93], [-1.0, 0.39]), d, side, EGO_PAINT.seam, 0.003));
    skin.push(skinPlate(pod, d, side, EGO_PAINT.trim));
  }

  // Normals came from the clean extrusion (extrudeProfile); bend them with it.
  deformGeometry(shell, d);
  deformGeometry(skirtGeo, d);

  /* ---- lamps ---- */
  const zFlat = d.hw - BEVEL; // where the nose and tail walls end
  const lit = (yRef: number, front: boolean) => {
    const y = Y(yRef);
    return { x: outlineX(body, y, front) + (front ? 0.004 : -0.004), y };
  };
  /** Two points round the bevelled corner, so the bar wraps onto the side. */
  const wrap = (x: number, y: number, front: boolean, side: 1 | -1) => {
    const back = front ? -1 : 1;
    return [
      new THREE.Vector3(x + back * 0.3 * BEVEL, y, side * (d.hw - 0.3 * BEVEL)),
      new THREE.Vector3(x + back * 0.72 * BEVEL, y, side * (d.hw - 0.04 * BEVEL)),
    ];
  };
  const span = (x: number, y: number, z0: number, z1: number, n: number) =>
    Array.from({ length: n + 1 }, (_, i) => new THREE.Vector3(x, y, z0 + ((z1 - z0) * i) / n));

  // Front: one thin full-width bar; its outer ends, round the corners, are
  // the indicators.
  const f = lit(0.6, true);
  const zInd = 0.74 * zFlat;
  const head = lampBar(span(f.x, f.y, -zInd, zInd, 12), d, 0.012, LENS.head);
  // Rear: a bar along the tail's upper edge and two short ones low on the bumper.
  const r = lit(0.92, false);
  const xLow = X(-2.345) - 0.004;
  const tail = merge([
    lampBar(span(r.x, r.y, -zInd, zInd, 12), d, 0.014, LENS.tail),
    lampBar(span(xLow, Y(0.4), 0.5 * zFlat, 0.86 * zFlat, 3), d, 0.011, LENS.tail),
    lampBar(span(xLow, Y(0.4), -0.86 * zFlat, -0.5 * zFlat, 3), d, 0.011, LENS.tail),
  ]);
  const marker = new THREE.Shape([
    new THREE.Vector2(X(1.93), Y(0.47)),
    new THREE.Vector2(X(2.03), Y(0.47)),
    new THREE.Vector2(X(2.03), Y(0.495)),
    new THREE.Vector2(X(1.93), Y(0.495)),
  ]);
  const indicator = (side: 1 | -1) =>
    merge([
      lampBar([...span(f.x, f.y, side * zInd, side * zFlat, 3), ...wrap(f.x, f.y, true, side)], d, 0.012, LENS.amber),
      lampBar([...span(r.x, r.y, side * zInd, side * zFlat, 3), ...wrap(r.x, r.y, false, side)], d, 0.014, LENS.amber),
      skinPlate(marker, d, side, LENS.amber, 0.004),
    ]);

  const tyreW = 0.235 * ky;
  return {
    shell: addFinish(merge([shell, skirtGeo, ...skin])),
    lamps: {
      head: merge([head]),
      tail,
      indicatorLeft: indicator(-1),
      indicatorRight: indicator(1),
    },
    wheel: { radius: wheelR, width: tyreW, axleX, centreZ: d.hw - tyreW / 2 - 0.02 },
    beam: deform(new THREE.Vector3(f.x - 0.12, f.y, 0.55 * zFlat), d),
  };
}

/**
 * One wheel: tyre, flat aero cover and centre cone, outer face toward +Z.
 * The cover carries five faint radial marks so the spin can be seen.
 */
export function buildEgoWheel(radius: number, width: number): THREE.BufferGeometry {
  const r = radius;
  const h = width / 2;
  const rim = 0.8 * r;
  const tyre = new THREE.LatheGeometry(
    [
      new THREE.Vector2(rim, -h),
      new THREE.Vector2(r - 0.035, -h),
      new THREE.Vector2(r, -h + 0.04),
      new THREE.Vector2(r, h - 0.04),
      new THREE.Vector2(r - 0.035, h),
      new THREE.Vector2(rim, h),
    ],
    32,
  ).rotateX(Math.PI / 2);
  const cover = new THREE.CylinderGeometry(rim + 0.004, rim + 0.004, 0.02, 40)
    .rotateX(Math.PI / 2)
    .translate(0, 0, h - 0.03);
  const back = new THREE.CylinderGeometry(rim, rim, 0.01, 24)
    .rotateX(Math.PI / 2)
    .translate(0, 0, -h + 0.02);
  const hub = new THREE.ConeGeometry(0.16 * r, 0.035, 32)
    .rotateX(Math.PI / 2)
    .translate(0, 0, h - 0.02 + 0.0175);
  const parts = [
    paint(tyre, EGO_PAINT.tyre),
    paint(cover, EGO_PAINT.cover),
    paint(back, EGO_PAINT.tyre),
    paint(hub, EGO_PAINT.cover),
  ];
  for (let k = 0; k < 5; k++) {
    const mark = new THREE.BoxGeometry(0.28 * r, 0.014, 0.004)
      .translate(0.55 * r, 0, h - 0.019)
      .rotateZ((k * 2 * Math.PI) / 5);
    parts.push(paint(mark, EGO_PAINT.coverMark));
  }
  return addFinish(merge(parts));
}

/** Reflection strength at full daylight. */
export const REFLECT = 1.15;

let reflections: THREE.DataTexture | null = null;

/** Deterministic 0..1 noise for the skyline. */
const hash = (n: number) => {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
  return x - Math.floor(x);
};

/**
 * A daylight street for the paint and glass to reflect: a bright sky that
 * deepens toward the zenith, a skyline of warm building blocks at the horizon
 * and grey asphalt below. The skyline is what gives the flanks the broken,
 * street-like reflections of the real car; the bright band above it rides
 * along the shoulders and shows every change of curvature.
 */
export function egoReflections(): THREE.DataTexture {
  if (reflections) return reflections;
  const w = 256;
  const h = 128;
  const data = new Float32Array(w * h * 4);
  const sky = (el: number): [number, number, number] => {
    const k = Math.pow(1 - el / (Math.PI / 2), 1.6);
    return [0.55 + 0.6 * k, 0.66 + 0.52 * k, 0.86 + 0.34 * k];
  };
  for (let j = 0; j < h; j++) {
    const el = ((j + 0.5) / h) * Math.PI - Math.PI / 2; // row 0 looks down
    for (let i = 0; i < w; i++) {
      const az = (i + 0.5) / w; // 0..1 around
      // Asphalt below, lightening toward the horizon, and sky above -- blended
      // over a wide band, never a step: a hard horizon, reflected through
      // interpolated normals, draws every triangle edge on a curved panel.
      const ground = 0.1 + 0.12 * Math.exp(Math.min(0, el) * 5);
      const s0 = sky(Math.max(0, el));
      const up = sm((el + 0.3) / 0.55);
      let c: [number, number, number] = [0, 1, 2].map((k) => ground + (s0[k] - ground) * up) as [
        number,
        number,
        number,
      ];
      {
        // Skyline: 10 blocks of varying height in brick and stone tones, a gap
        // here and there. Edges are cross-faded, never hard: on a curved panel
        // a hard edge in the reflection reads as a crease or a smear.
        const f = az * 10;
        const blk = Math.floor(f);
        const mixB = sm((f - blk - 0.5) / 0.5); // blend into the next block
        const bldg = (n: number): [number, number] => {
          const top = hash(n) < 0.18 ? 0 : 0.08 + 0.28 * hash(n + 50);
          return [top, hash(n + 99) < 0.5 ? 0 : 1];
        };
        const [t0, k0] = bldg(blk);
        const [t1, k1] = bldg(blk + 1);
        const top = t0 + (t1 - t0) * mixB;
        const stone = k0 + (k1 - k0) * mixB;
        // Fades out below the horizon too, into the street.
        const cover = 0.6 * (1 - sm((el - top + 0.08) / 0.16)) * sm((el + 0.3) / 0.3);
        const wall: [number, number, number] = [
          0.34 + 0.06 * stone,
          0.22 + 0.13 * stone,
          0.17 + 0.15 * stone,
        ];
        c = [0, 1, 2].map((k) => c[k] + (wall[k] - c[k]) * cover) as [number, number, number];
      }
      data.set([c[0], c[1], c[2], 1], (j * w + i) * 4);
    }
  }
  reflections = new THREE.DataTexture(data, w, h, THREE.RGBAFormat, THREE.FloatType);
  reflections.mapping = THREE.EquirectangularReflectionMapping;
  reflections.colorSpace = THREE.LinearSRGBColorSpace;
  reflections.magFilter = THREE.LinearFilter;
  reflections.needsUpdate = true;
  return reflections;
}

/**
 * One material for the shell and the wheels; each vertex carries its own
 * finish (see FINISH), so metallic paint, glass and rubber share a draw call.
 */
export function egoMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial();
  const finish = attribute<'vec2'>('finish', 'vec2');
  mat.colorNode = attribute<'vec3'>('color', 'vec3');
  mat.roughnessNode = finish.x;
  mat.metalnessNode = finish.y;
  mat.envMap = egoReflections();
  mat.envMapIntensity = REFLECT;
  return mat;
}
