/**
 * Stylised vehicle meshes.
 *
 * A whole car — body, greenhouse, wheels, lamps — is merged into a single
 * vertex-coloured buffer, so every vehicle in the scene costs exactly one draw
 * call. Roughness is derived from colour in TSL: the dark glass and tyres come
 * out glossier than the painted body without needing a second material.
 */
import * as THREE from 'three/webgpu';
import { attribute, float, mix } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import type { Pose, Size } from '../schema';

export interface VehicleStyle {
  body: string;
  glass: string;
  wheel: string;
  headlight: string;
  taillight: string;
  /** Boxy silhouette for trucks and buses. */
  boxy?: boolean;
}

export const EGO_STYLE: VehicleStyle = {
  body: '#FFFFFF',
  glass: '#2B3743',
  wheel: '#23282E',
  headlight: '#EAF6FF',
  taillight: '#E5484D',
};

export const TRAFFIC_STYLES: Record<string, VehicleStyle> = {
  car: { body: '#8FA3B8', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F' },
  truck: { body: '#9C8FBC', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F', boxy: true },
  bus: { body: '#B9A05F', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F', boxy: true },
  motorcycle: { body: '#5E93A8', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F' },
  cyclist: { body: '#5E93A8', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F' },
  pedestrian: { body: '#D4A03C', glass: '#8A6A28', wheel: '#8A6A28', headlight: '#F5E4C0', taillight: '#D4A03C' },
  unknown: { body: '#A6B1BD', glass: '#39434F', wheel: '#23282E', headlight: '#E8F0F7', taillight: '#C0453F' },
};

/* ------------------------------------------------------------------ */
/* Geometry                                                            */
/* ------------------------------------------------------------------ */

/** Centred rounded rectangle in the XY plane; +X is the vehicle's forward. */
function roundedRectShape(length: number, width: number, r: number): THREE.Shape {
  const hx = length / 2;
  const hy = width / 2;
  const rad = Math.min(r, hx * 0.9, hy * 0.9);
  const s = new THREE.Shape();
  s.moveTo(-hx + rad, -hy);
  s.lineTo(hx - rad, -hy);
  s.quadraticCurveTo(hx, -hy, hx, -hy + rad);
  s.lineTo(hx, hy - rad);
  s.quadraticCurveTo(hx, hy, hx - rad, hy);
  s.lineTo(-hx + rad, hy);
  s.quadraticCurveTo(-hx, hy, -hx, hy - rad);
  s.lineTo(-hx, -hy + rad);
  s.quadraticCurveTo(-hx, -hy, -hx + rad, -hy);
  return s;
}

function paint(geo: THREE.BufferGeometry, hex: string): THREE.BufferGeometry {
  const g = geo.index ? geo.toNonIndexed() : geo;
  if (g !== geo) geo.dispose();
  g.clearGroups();
  g.deleteAttribute('uv2');
  const c = new THREE.Color(hex).convertSRGBToLinear();
  const n = g.attributes.position.count;
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  if (!g.attributes.uv) {
    g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(n * 2), 2));
  }
  return g;
}

/** Plan-view slab, extruded upward with soft edges. */
function slab(
  length: number,
  width: number,
  cornerR: number,
  height: number,
  bevel: number,
): THREE.BufferGeometry {
  const geo = new THREE.ExtrudeGeometry(roundedRectShape(length, width, cornerR), {
    depth: Math.max(0.02, height - bevel * 2),
    bevelEnabled: bevel > 0,
    bevelThickness: bevel,
    bevelSize: bevel,
    bevelSegments: 2,
    curveSegments: 4,
  });
  // Shape XY -> ground plane, extrusion -> +Y.
  geo.rotateX(-Math.PI / 2);
  geo.translate(0, bevel, 0);
  return geo;
}

/**
 * Build a complete vehicle as one buffer. Origin is at the centre of the
 * footprint on the ground; +X is forward, +Y up.
 */
export function buildVehicleGeometry(
  size: Size,
  style: VehicleStyle,
): THREE.BufferGeometry {
  const { length: L, width: W, height: H } = size;
  const parts: THREE.BufferGeometry[] = [];

  const wheelR = Math.min(0.36, H * 0.26);
  const bodyBottom = wheelR * 0.62;

  if (style.boxy) {
    // Cab + box body, for trucks and buses.
    const cabL = L * 0.3;
    const cab = slab(cabL, W * 0.98, 0.32, H * 0.72, 0.07);
    cab.translate(L / 2 - cabL / 2, bodyBottom, 0);
    parts.push(paint(cab, style.body));

    const boxL = L * 0.66;
    const box = slab(boxL, W, 0.16, H - bodyBottom - 0.05, 0.05);
    box.translate(-L / 2 + boxL / 2, bodyBottom, 0);
    parts.push(paint(box, style.body));

    const glass = slab(cabL * 0.5, W * 0.86, 0.16, 0.02, 0);
    glass.translate(L / 2 - cabL * 0.34, H * 0.62, 0);
    parts.push(paint(glass, style.glass));
  } else {
    const bodyH = H * 0.46;
    const body = slab(L, W, W * 0.34, bodyH, 0.09);
    body.translate(0, bodyBottom, 0);
    parts.push(paint(body, style.body));

    // Greenhouse, set back and inboard so the shoulder line reads.
    const cabL = L * 0.5;
    const cab = slab(cabL, W * 0.82, W * 0.3, H - bodyBottom - bodyH, 0.11);
    cab.translate(-L * 0.045, bodyBottom + bodyH - 0.02, 0);
    parts.push(paint(cab, style.body));

    // Glass band around the greenhouse.
    const glassH = (H - bodyBottom - bodyH) * 0.56;
    const glass = slab(cabL * 0.96, W * 0.845, W * 0.3, glassH, 0.05);
    glass.translate(-L * 0.045, bodyBottom + bodyH + 0.06, 0);
    parts.push(paint(glass, style.glass));
  }

  // Wheels: cylinder axis along Z (the vehicle's lateral axis).
  const wheelGeo = new THREE.CylinderGeometry(wheelR, wheelR, W * 0.12, 14);
  wheelGeo.rotateX(Math.PI / 2);
  const axleX = L * 0.31;
  const axleZ = W / 2 - W * 0.055;
  for (const sx of [1, -1]) {
    for (const sz of [1, -1]) {
      const w = wheelGeo.clone();
      w.translate(sx * axleX, wheelR, sz * axleZ);
      parts.push(paint(w, style.wheel));
    }
  }
  wheelGeo.dispose();

  // Lamp strips.
  const lampGeo = new THREE.BoxGeometry(0.07, 0.1, W * 0.26);
  for (const sz of [1, -1]) {
    const head = lampGeo.clone();
    head.translate(L / 2 - 0.03, bodyBottom + H * 0.24, sz * W * 0.3);
    parts.push(paint(head, style.headlight));
    const tail = lampGeo.clone();
    tail.translate(-L / 2 + 0.03, bodyBottom + H * 0.28, sz * W * 0.3);
    parts.push(paint(tail, style.taillight));
  }
  lampGeo.dispose();

  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  if (!merged) throw new Error('failed to merge vehicle geometry');
  merged.computeBoundingSphere();
  return merged;
}

/**
 * One material for every vehicle. Roughness falls with luminance so glass and
 * tyres read as glossy while painted panels stay satin.
 */
export function vehicleMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0.08 });
  const base = attribute<'vec3'>('color', 'vec3');
  const lum = base.r.mul(0.299).add(base.g.mul(0.587)).add(base.b.mul(0.114));
  mat.colorNode = base;
  mat.roughnessNode = mix(float(0.16), float(0.46), lum.clamp(0, 1));
  return mat;
}

/* ------------------------------------------------------------------ */
/* Ego vehicle                                                         */
/* ------------------------------------------------------------------ */

export class EgoVehicle {
  readonly group: THREE.Group;
  readonly mesh: THREE.Mesh;
  private readonly geometry: THREE.BufferGeometry;
  private readonly material: THREE.MeshStandardNodeMaterial;

  constructor(size: Size) {
    this.geometry = buildVehicleGeometry(size, EGO_STYLE);
    this.material = vehicleMaterial();
    this.mesh = new THREE.Mesh(this.geometry, this.material);
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group = new THREE.Group();
    this.group.name = 'ego';
    this.group.add(this.mesh);
  }

  /** World pose -> three.js transform. Heading maps straight to rotation.y. */
  setPose(pose: Pose): void {
    this.group.position.set(pose.x, 0, -pose.y);
    this.group.rotation.y = pose.heading;
  }

  /** Subtle body roll and pitch, driven by steering and acceleration. */
  setAttitude(steering: number, accel: number): void {
    this.mesh.rotation.x = THREE.MathUtils.clamp(accel * 0.012, -0.03, 0.03);
    this.mesh.rotation.z = THREE.MathUtils.clamp(-steering * 0.05, -0.04, 0.04);
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
  }
}
