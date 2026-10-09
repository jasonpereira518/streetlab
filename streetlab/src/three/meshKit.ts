/**
 * Shared building blocks for the stylised agent meshes (vehicles, riders,
 * pedestrians): vertex-coloured geometry with a per-vertex `emit` weight, and
 * the one material that reads both.
 *
 * Everything here is deterministic -- no Math.random -- because the renderer's
 * pictures are what the detector is scored on, and a scene must look the same
 * every time it is built.
 */
import * as THREE from 'three/webgpu';
import { attribute, float, mix, positionLocal, smoothstep } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/** Stable 32-bit hash (FNV-1a) so an agent id picks the same paint every run. */
export function hashId(id: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < id.length; i++) {
    h ^= id.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/**
 * Bake a flat colour (and an emissive weight, 0..1) into a geometry. Normalises
 * the attribute set so any two painted parts can be merged.
 */
export function paint(geo: THREE.BufferGeometry, hex: string, emit = 0): THREE.BufferGeometry {
  const g = geo.index ? geo.toNonIndexed() : geo;
  if (g !== geo) geo.dispose();
  g.clearGroups();
  g.deleteAttribute('uv2');
  const c = new THREE.Color(hex).convertSRGBToLinear();
  const n = g.attributes.position.count;
  const colors = new Float32Array(n * 3);
  const emits = new Float32Array(n).fill(emit);
  for (let i = 0; i < n; i++) {
    colors[i * 3] = c.r;
    colors[i * 3 + 1] = c.g;
    colors[i * 3 + 2] = c.b;
  }
  g.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  g.setAttribute('emit', new THREE.BufferAttribute(emits, 1));
  if (!g.attributes.uv) {
    g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(n * 2), 2));
  }
  return g;
}

/** Axis-aligned box centred on (x, y, z); +X forward, +Y up. */
export function boxAt(
  l: number,
  h: number,
  w: number,
  x: number,
  y: number,
  z: number,
): THREE.BufferGeometry {
  return new THREE.BoxGeometry(l, h, w).translate(x, y, z);
}

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

/**
 * Plan-view slab with its footprint centred on the origin, extruded upward from
 * y = 0 to `height`, with soft edges.
 */
export function slab(
  length: number,
  width: number,
  cornerR: number,
  height: number,
  bevel: number,
): THREE.BufferGeometry {
  // The bevel grows the outline by `bevel` on every side; shrink the shape so
  // the finished footprint is exactly `length` x `width` (labels are sized from it).
  const geo = new THREE.ExtrudeGeometry(roundedRectShape(length - 2 * bevel, width - 2 * bevel, cornerR), {
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

/** Wheel: a cylinder whose axis runs along Z (the vehicle's lateral axis). */
export function wheelAt(
  radius: number,
  width: number,
  x: number,
  y: number,
  z: number,
): THREE.BufferGeometry {
  const g = new THREE.CylinderGeometry(radius, radius, width, 16);
  g.rotateX(Math.PI / 2);
  return g.translate(x, y, z);
}

/** Merge painted parts into one buffer and release the inputs. */
export function mergeParts(parts: THREE.BufferGeometry[]): THREE.BufferGeometry {
  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  if (!merged) throw new Error('failed to merge mesh parts');
  merged.computeBoundingSphere();
  return merged;
}

/**
 * One material for every agent. Roughness falls with luminance so glass and
 * tyres read as glossy while painted panels stay satin; lamps glow through
 * `emit` regardless of the key light.
 */
export function agentMaterial(): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0.08 });
  const base = attribute<'vec3'>('color', 'vec3');
  const emit = attribute<'float'>('emit', 'float');
  const lum = base.r.mul(0.299).add(base.g.mul(0.587)).add(base.b.mul(0.114));
  // Fake ambient occlusion: the lower third of every agent darkens, which is
  // most of what separates a solid object from a flat decal at this size.
  const ao = mix(float(0.8), float(1.0), smoothstep(float(0.05), float(0.7), positionLocal.y));
  mat.colorNode = base.mul(ao);
  mat.roughnessNode = mix(float(0.16), float(0.46), lum.clamp(0, 1));
  mat.emissiveNode = base.mul(emit.mul(1.6));
  return mat;
}
