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
import { attitudeOn, type HeightFn } from './terrain';
import { LENS, REFLECT, buildEgoBody, buildEgoWheel, egoMaterial } from './egoBody';

export interface VehicleStyle {
  body: string;
  glass: string;
  wheel: string;
  headlight: string;
  taillight: string;
  /** Boxy silhouette for trucks and buses. */
  boxy?: boolean;
}

/** The ego's paint, for the parts drawn outside egoBody.ts (the cabin's bonnet). */
export const EGO_STYLE: VehicleStyle = {
  body: '#C8B288',
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

export function paint(geo: THREE.BufferGeometry, hex: string): THREE.BufferGeometry {
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
export function slab(
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

export type TurnSignal = 'off' | 'left' | 'right';

/** Indicator cadence: 1.5 Hz, half on, half off. */
export const BLINK_HZ = 1.5;
export const blinkOn = (t: number): boolean => (t * BLINK_HZ) % 1 < 0.5;

/** Above this pedal the brake lights come on. */
const BRAKE_ON = 0.05;

const LAMP = {
  head: { glow: '#EAF4FF', on: 2.4 },
  tail: { glow: '#FF2A2A', running: 0.5, brake: 2.6 },
  amber: { glow: '#FFA21A', on: 2.6 },
} as const;

function lampMaterial(lens: string, glow: string): THREE.MeshStandardNodeMaterial {
  return new THREE.MeshStandardNodeMaterial({
    color: lens,
    emissive: glow,
    emissiveIntensity: 0,
    roughness: 0.25,
  });
}

interface Wheel {
  /** Steers about Y; at the hub. */
  pivot: THREE.Group;
  /** Rolls about Z. */
  spin: THREE.Group;
  front: boolean;
}

/**
 * The ego car: a painted shell, four wheels that roll and steer, light bars
 * that glow, and two headlight beams.
 *
 * The wheels hang off `group`, not the body `mesh`, so the body's roll and
 * pitch (`setAttitude`) move over them like a sprung body over its wheels.
 * Lamps and beams ride on `mesh`.
 */
export class EgoVehicle {
  readonly group: THREE.Group;
  /** The body. The cabin is parented to it; it survives `setSize`. */
  readonly mesh: THREE.Mesh;
  /** Headlight beams. Always in the scene; switched by intensity, because
   * adding or removing a light recompiles every material. */
  readonly beams: THREE.SpotLight[] = [];

  private size: Size;
  private readonly material = egoMaterial();
  private wheelGeometry: THREE.BufferGeometry | null = null;
  private wheelRadius = 0.37;
  private readonly wheels: Wheel[] = [];
  private readonly lamps: Record<'head' | 'tail' | 'indicatorLeft' | 'indicatorRight', THREE.Mesh>;
  private readonly lampMats = {
    head: lampMaterial(LENS.head, LAMP.head.glow),
    tail: lampMaterial(LENS.tail, LAMP.tail.glow),
    indicatorLeft: lampMaterial(LENS.amber, LAMP.amber.glow),
    indicatorRight: lampMaterial(LENS.amber, LAMP.amber.glow),
  };
  private headlights = false;

  constructor(size: Size) {
    this.size = size;
    this.group = new THREE.Group();
    this.group.name = 'ego';
    this.mesh = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
    this.mesh.name = 'ego-body';
    this.mesh.castShadow = true;
    this.mesh.receiveShadow = true;
    this.group.add(this.mesh);

    this.lamps = {
      head: new THREE.Mesh(new THREE.BufferGeometry(), this.lampMats.head),
      tail: new THREE.Mesh(new THREE.BufferGeometry(), this.lampMats.tail),
      indicatorLeft: new THREE.Mesh(new THREE.BufferGeometry(), this.lampMats.indicatorLeft),
      indicatorRight: new THREE.Mesh(new THREE.BufferGeometry(), this.lampMats.indicatorRight),
    };
    for (const [name, lamp] of Object.entries(this.lamps)) {
      lamp.name = `ego-lamp-${name}`;
      this.mesh.add(lamp);
    }

    for (let i = 0; i < 4; i++) {
      const pivot = new THREE.Group();
      const spin = new THREE.Group();
      const tyre = new THREE.Mesh(new THREE.BufferGeometry(), this.material);
      tyre.name = 'ego-wheel';
      tyre.castShadow = true;
      // Built with its face toward +Z; the left pair turn it outward.
      if (i % 2 === 1) tyre.rotation.y = Math.PI;
      spin.add(tyre);
      pivot.add(spin);
      this.group.add(pivot);
      this.wheels.push({ pivot, spin, front: i < 2 });
    }

    for (const side of [1, -1]) {
      const beam = new THREE.SpotLight('#F2F6FF', 0, 45, 0.45, 0.6, 2);
      beam.name = 'ego-beam';
      beam.castShadow = false;
      beam.userData.side = side;
      this.mesh.add(beam, beam.target);
      this.beams.push(beam);
    }

    this.build(size);
  }

  /** Rebuild the geometry for a new footprint; a no-op if it is unchanged. */
  setSize(size: Size): void {
    const s = this.size;
    if (s.length === size.length && s.width === size.width && s.height === size.height) return;
    this.size = size;
    this.build(size);
  }

  private build(size: Size): void {
    const body = buildEgoBody(size);
    this.mesh.geometry.dispose();
    this.mesh.geometry = body.shell;
    for (const key of Object.keys(this.lamps) as Array<keyof typeof this.lamps>) {
      this.lamps[key].geometry.dispose();
      this.lamps[key].geometry = body.lamps[key];
    }

    const { radius, width, axleX, centreZ } = body.wheel;
    this.wheelRadius = radius;
    this.wheelGeometry?.dispose();
    this.wheelGeometry = buildEgoWheel(radius, width);
    this.wheels.forEach((w, i) => {
      // 0: front right, 1: front left, 2: rear right, 3: rear left (-Z is left).
      w.pivot.position.set(w.front ? axleX : -axleX, radius, i % 2 === 0 ? centreZ : -centreZ);
      (w.spin.children[0] as THREE.Mesh).geometry = this.wheelGeometry!;
    });

    for (const beam of this.beams) {
      const side = beam.userData.side as number;
      beam.position.set(body.beam.x, body.beam.y, side * body.beam.z);
      // Dipped: down onto the road ~20 m ahead.
      beam.target.position.set(body.beam.x + 20, 0, side * body.beam.z * 1.6);
    }
  }

  /** World pose -> three.js transform. Heading maps straight to rotation.y. */
  setPose(pose: Pose, ground: HeightFn | null = null): void {
    const a = attitudeOn(ground, pose.x, pose.y, pose.heading, this.size.length, this.size.width);
    this.group.position.set(pose.x, a.y, -pose.y);
    this.group.rotation.order = 'YZX';
    this.group.rotation.set(a.roll, pose.heading, a.pitch);
  }

  /**
   * Subtle body pitch and roll: the nose lifts under throttle and dips under
   * braking (about Z), and the body leans out of a turn (about X).
   */
  setAttitude(steering: number, accel: number): void {
    this.mesh.rotation.z = THREE.MathUtils.clamp(accel * 0.012, -0.03, 0.03);
    this.mesh.rotation.x = THREE.MathUtils.clamp(steering * 0.05, -0.04, 0.04);
  }

  /**
   * Scale the paint's reflections with the scene's light, 0..1 of daylight.
   * Squared: the reflected street is lit by the same light, so at night (about
   * a third of the daylight ambient) it should be near-black, not a third.
   */
  setReflections(k: number): void {
    this.material.envMapIntensity = REFLECT * THREE.MathUtils.clamp(k, 0, 1) ** 2;
  }

  setHeadlights(on: boolean): void {
    this.headlights = on;
    for (const beam of this.beams) beam.intensity = on ? 60 : 0;
    this.lampMats.head.emissiveIntensity = on ? LAMP.head.on : 0;
  }

  get headlightsOn(): boolean {
    return this.headlights;
  }

  /**
   * Per-frame animation. `speed` m/s, `steering` the road-wheel angle (rad,
   * + left), `brake` the pedal 0..1, `t` a wall clock in seconds for the
   * blinker phase.
   */
  update(speed: number, steering: number, brake: number, signal: TurnSignal, t: number, dt: number): void {
    // Rolling forward (+X) turns the wheel clockwise seen from +Z.
    const roll = (speed / this.wheelRadius) * dt;
    for (const w of this.wheels) {
      w.spin.rotation.z = (w.spin.rotation.z - roll) % (2 * Math.PI);
      w.pivot.rotation.y = w.front ? steering : 0;
    }

    this.lampMats.tail.emissiveIntensity =
      brake > BRAKE_ON ? LAMP.tail.brake : this.headlights ? LAMP.tail.running : 0;
    const blink = signal !== 'off' && blinkOn(t);
    this.lampMats.indicatorLeft.emissiveIntensity = blink && signal === 'left' ? LAMP.amber.on : 0;
    this.lampMats.indicatorRight.emissiveIntensity = blink && signal === 'right' ? LAMP.amber.on : 0;
  }

  dispose(): void {
    this.mesh.geometry.dispose();
    for (const lamp of Object.values(this.lamps)) lamp.geometry.dispose();
    this.wheelGeometry?.dispose();
    this.material.dispose();
    for (const m of Object.values(this.lampMats)) m.dispose();
    for (const beam of this.beams) beam.dispose();
  }
}
