/**
 * The ego's interior, seen from the driver's seat.
 *
 * Built procedurally like the exterior (ego.ts): every static panel is merged
 * into one vertex-coloured buffer; only the parts that move -- the steering
 * wheel, the indicator stalk -- and the centre screen are separate meshes.
 *
 * Frame: the same as the ego mesh it is parented to. Origin at the centre of
 * the footprint on the ground, +X forward, +Y up, -Z is the car's left (sim +y
 * maps to three -z). Left-hand drive, so the driver sits at -Z.
 *
 * The whole cabin lives on CH.CABIN: the main camera shows it only in cockpit
 * view, and neither the detector nor the centre screen's own camera ever sees
 * it (see layers.ts).
 */
import * as THREE from 'three/webgpu';
import { attribute, float } from 'three/tsl';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';
import { DEG_TO_RAD, clamp, damp, handWheelDeg } from '../units';
import { EGO_STYLE, paint, slab } from './ego';
import { CH, assignChannel } from './layers';

export type TurnSignal = 'off' | 'left' | 'right';

const TRIM = '#43474D';
const TRIM_DARK = '#2C2F34';
const HEADLINER = '#B9B6B0';
const WOOD = '#8C7A63';
const PILLAR = '#9C9993';

/** Driver's eye, at rest, in the cabin frame. */
export const EYE = { x: -0.25, y: 1.18, z: -0.37, pitch: -0.06 } as const;
/** Steering-wheel hub and rim. */
const WHEEL = { x: 0.2, y: 0.93, z: EYE.z, r: 0.18, tilt: -0.35 } as const;
/** Centre screen, landscape, ~15 inch. */
export const SCREEN = { x: 0.5, y: 0.99, z: 0.02, w: 0.34, h: 0.21, tilt: 0.18 } as const;

/** Indicator cadence: 1.5 Hz, half on, half off. */
export const BLINK_HZ = 1.5;
export const blinkOn = (t: number): boolean => (t * BLINK_HZ) % 1 < 0.5;

/** A box spanning two points, `w` x `h` in cross-section. */
function beam(a: THREE.Vector3, b: THREE.Vector3, w: number, h: number): THREE.BufferGeometry {
  const len = a.distanceTo(b);
  const g = new THREE.BoxGeometry(len, h, w);
  const dir = b.clone().sub(a).normalize();
  g.applyQuaternion(new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(1, 0, 0), dir));
  g.translate((a.x + b.x) / 2, (a.y + b.y) / 2, (a.z + b.z) / 2);
  return g;
}

const box = (lx: number, ly: number, lz: number, x: number, y: number, z: number) =>
  new THREE.BoxGeometry(lx, ly, lz).translate(x, y, z);

function shellGeometry(): THREE.BufferGeometry {
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  const parts: THREE.BufferGeometry[] = [];

  // Hood, seen through the windscreen: the exterior shell is hidden in this
  // view, so the cabin carries its own. Falls away slightly toward the nose.
  const hood = slab(1.5, 1.86, 0.5, 0.05, 0.02).rotateZ(-0.07).translate(1.72, 0.84, 0);
  parts.push(paint(hood, EGO_STYLE.body));

  // Dash: one soft block, a wood strip across its face, a vent slit above it.
  parts.push(paint(slab(0.62, 1.74, 0.12, 0.3, 0.05).translate(0.74, 0.6, 0), TRIM));
  parts.push(paint(box(0.02, 0.05, 1.6, 0.43, 0.8, 0), WOOD));
  parts.push(paint(box(0.02, 0.012, 1.5, 0.432, 0.845, 0), TRIM_DARK));
  // Cowl under the windscreen base.
  parts.push(paint(box(0.12, 0.03, 1.7, 1.02, 0.9, 0), TRIM_DARK));

  // Windscreen frame: A-pillars, header, roof.
  for (const s of [-1, 1]) {
    parts.push(paint(beam(v(1.04, 0.9, s * 0.82), v(0.34, 1.36, s * 0.74), 0.07, 0.09), PILLAR));
    parts.push(paint(beam(v(0.34, 1.37, s * 0.74), v(-1.3, 1.37, s * 0.76), 0.06, 0.06), PILLAR));
    // B-pillar.
    parts.push(paint(beam(v(-0.6, 0.93, s * 0.83), v(-0.62, 1.37, s * 0.77), 0.08, 0.1), PILLAR));
    // Door card, top at the beltline.
    parts.push(paint(box(2.3, 0.45, 0.06, -0.15, 0.71, s * 0.86), TRIM));
    parts.push(paint(box(2.3, 0.03, 0.12, -0.15, 0.935, s * 0.83), TRIM_DARK));
    // Wing mirror outside the door: body-coloured housing, dark glass facing back.
    parts.push(paint(box(0.12, 0.11, 0.18, 0.82, 1.0, s * 1.04), EGO_STYLE.body));
    parts.push(paint(box(0.01, 0.09, 0.16, 0.755, 1.0, s * 1.04), EGO_STYLE.glass));
  }
  parts.push(paint(box(0.09, 0.05, 1.5, 0.34, 1.37, 0), PILLAR));
  parts.push(paint(box(1.64, 0.02, 1.5, -0.48, 1.39, 0), HEADLINER));

  // Rear-view mirror on a stem from the header.
  parts.push(paint(box(0.02, 0.06, 0.02, 0.42, 1.33, 0), TRIM_DARK));
  parts.push(paint(box(0.04, 0.06, 0.22, 0.41, 1.29, 0), TRIM_DARK));

  // Steering column into the dash.
  const hub = v(WHEEL.x, WHEEL.y, WHEEL.z);
  const toDash = v(Math.cos(WHEEL.tilt), Math.sin(WHEEL.tilt), 0).multiplyScalar(0.3);
  parts.push(paint(beam(hub, hub.clone().add(toDash), 0.06, 0.06), TRIM_DARK));

  // Centre-screen bezel; the screen itself is a separate mesh.
  const bezel = box(0.02, SCREEN.h + 0.02, SCREEN.w + 0.02, 0, 0, 0)
    .rotateZ(SCREEN.tilt)
    .translate(SCREEN.x + 0.012, SCREEN.y, SCREEN.z);
  parts.push(paint(bezel, TRIM_DARK));

  const merged = mergeGeometries(parts, false);
  for (const p of parts) p.dispose();
  if (!merged) throw new Error('failed to merge cabin geometry');
  merged.computeBoundingSphere();
  return merged;
}

/** Rim, three spokes and a hub, in a plane facing -X; rotates about X. */
function wheelGeometry(): THREE.BufferGeometry {
  const parts: THREE.BufferGeometry[] = [];
  parts.push(paint(new THREE.TorusGeometry(WHEEL.r, 0.017, 10, 40).rotateY(Math.PI / 2), '#34383D'));
  parts.push(paint(new THREE.CylinderGeometry(0.05, 0.055, 0.04, 20).rotateZ(Math.PI / 2), TRIM_DARK));
  // Spokes at 3, 9 and 6 o'clock as the driver sees them.
  for (const a of [0, Math.PI, -Math.PI / 2]) {
    const s = box(0.015, 0.03, WHEEL.r - 0.04, 0, 0, 0)
      .translate(0, 0, (WHEEL.r + 0.04) / 2 - 0.02)
      .rotateX(a + Math.PI / 2);
    parts.push(paint(s, '#2A2E33'));
  }
  // A top-dead-centre stripe on the rim, so how far the wheel is turned
  // reads at a glance: a short, slightly fatter arc centred at 12 o'clock.
  const stripeArc = 0.2;
  parts.push(
    paint(
      new THREE.TorusGeometry(WHEEL.r, 0.0195, 10, 6, stripeArc)
        .rotateZ(Math.PI / 2 - stripeArc / 2)
        .rotateY(Math.PI / 2),
      '#E8E4DC',
    ),
  );
  const merged = mergeGeometries(parts, false)!;
  for (const p of parts) p.dispose();
  return merged;
}

function interiorMaterial(roughness: number): THREE.MeshStandardNodeMaterial {
  const mat = new THREE.MeshStandardNodeMaterial({ metalness: 0.02 });
  mat.colorNode = attribute<'vec3'>('color', 'vec3');
  mat.roughnessNode = float(roughness);
  return mat;
}

export class Cabin {
  readonly group = new THREE.Group();
  /** Where the cockpit camera sits; moves with the driver's head. */
  readonly eyeAnchor = new THREE.Object3D();
  /** The screen surface; fsdScreen.ts gives it a texture. */
  readonly screen: THREE.Mesh<THREE.PlaneGeometry, THREE.MeshBasicNodeMaterial>;
  /** Rotates about its own X axis (the column); its parent carries the tilt. */
  readonly wheel: THREE.Mesh;
  readonly stalk: THREE.Mesh;

  private readonly shellMat = interiorMaterial(0.85);
  private readonly wheelMat = interiorMaterial(0.45);
  /** Damped head offsets: surge (m, + forward), sway (m, + right), roll (rad). */
  private surge = 0;
  private sway = 0;
  private roll = 0;

  constructor() {
    this.group.name = 'cabin';
    const shell = new THREE.Mesh(shellGeometry(), this.shellMat);
    // Lit by the sky and sun, but kept out of the shadow map: the exterior
    // shell around it would otherwise black it out entirely.
    shell.receiveShadow = false;

    const column = new THREE.Group();
    column.position.set(WHEEL.x, WHEEL.y, WHEEL.z);
    column.rotation.z = WHEEL.tilt;
    this.wheel = new THREE.Mesh(wheelGeometry(), this.wheelMat);
    column.add(this.wheel);

    // Indicator stalk, left of the column: down for left, up for right.
    this.stalk = new THREE.Mesh(paint(box(0.03, 0.015, 0.11, 0, 0, -0.055), TRIM_DARK), this.shellMat);
    this.stalk.position.set(0.06, 0, -0.04);
    column.add(this.stalk);

    this.screen = new THREE.Mesh(
      new THREE.PlaneGeometry(SCREEN.w, SCREEN.h).rotateY(-Math.PI / 2),
      new THREE.MeshBasicNodeMaterial({ color: '#0B0F14', toneMapped: false }),
    );
    this.screen.position.set(SCREEN.x, SCREEN.y, SCREEN.z);
    this.screen.rotation.z = SCREEN.tilt;

    this.eyeAnchor.rotation.order = 'YXZ';
    this.eyeAnchor.rotation.set(EYE.pitch, -Math.PI / 2, 0);
    this.eyeAnchor.position.set(EYE.x, EYE.y, EYE.z);

    this.group.add(shell, column, this.screen, this.eyeAnchor);
    assignChannel(this.group, CH.CABIN);
  }

  /**
   * Animate from the car's state. `steering` is the road-wheel angle (rad,
   * + left), `accel` longitudinal m/s^2, `t` a wall clock in seconds for the
   * blinker phase.
   */
  update(
    steering: number,
    accel: number,
    speed: number,
    signal: TurnSignal,
    t: number,
    dt: number,
  ): void {
    // Left is counter-clockwise as the driver sees it: a negative turn about
    // the column, which points away from them (+X).
    this.wheel.rotation.x = -handWheelDeg(steering) * DEG_TO_RAD;

    this.stalk.rotation.y = signal === 'left' ? -0.18 : signal === 'right' ? 0.18 : 0;
    this.stalk.rotation.x = signal === 'left' ? 0.25 : signal === 'right' ? -0.25 : 0;

    // The body is thrown against the car's acceleration: back under throttle,
    // forward under braking, outward in a corner. Lateral g from the bicycle
    // model (wheelbase 2.9 m, as in sim/vehicle.py).
    const lat = (speed * speed * Math.tan(steering)) / 2.9;
    const g = 9.81;
    this.surge = damp(this.surge, clamp((-accel / g) * 0.06, -0.04, 0.04), 0.02, dt);
    this.sway = damp(this.sway, clamp((lat / g) * 0.05, -0.03, 0.03), 0.02, dt);
    this.roll = damp(this.roll, clamp((lat / g) * 0.06, -0.035, 0.035), 0.02, dt);
    // A faint tremor from the road surface, proportional to speed.
    const bob = 0.0015 * clamp(speed / 20, 0, 1) * Math.sin(t * 2 * Math.PI * 6.3);

    this.eyeAnchor.position.set(EYE.x + this.surge, EYE.y + bob, EYE.z + this.sway);
    // Braking nods the head down a touch; a corner leans it outward.
    this.eyeAnchor.rotation.x = EYE.pitch - this.surge * 0.4;
    this.eyeAnchor.rotation.z = this.roll;
  }

  dispose(): void {
    this.group.traverse((o) => {
      if (o instanceof THREE.Mesh) o.geometry.dispose();
    });
    this.shellMat.dispose();
    this.wheelMat.dispose();
    this.screen.material.dispose();
  }
}
