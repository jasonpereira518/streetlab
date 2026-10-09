/**
 * People and two-wheelers.
 *
 * A pedestrian used to be drawn as a 0.6 x 0.6 x 1.75 m vehicle with wheels, so
 * nothing a person detector knows about people could ever fire on it. These are
 * humanoids (head, torso, arms, legs, shoes) and a bicycle with a pedalling
 * rider, plus a motorcycle with a rider, all in the same one-material,
 * vertex-coloured style as the vehicles.
 *
 * Local frame matches the vehicles: origin on the ground at the footprint
 * centre, +X forward, +Y up, Z lateral. Reference heights are the sizes the
 * simulator spawns (pedestrian 1.75 m, cyclist 1.7 m, motorcycle 1.3 m).
 */
import * as THREE from 'three/webgpu';
import { boxAt, hashId, mergeParts, paint, wheelAt } from './meshKit';

const SKIN = ['#F1C9A5', '#D9A57E', '#B9835C', '#8D5A3C', '#5E3B27'];
const HAIR = ['#1B1713', '#3A2A1C', '#6B4A2B', '#A0814F', '#7A7A7A'];
const SHIRT = ['#C8372D', '#2E5FA3', '#E3B23C', '#3E8A5B', '#F2F2EE', '#6C4F9C', '#E07B39', '#2B2F36'];
const PANTS = ['#2B3A55', '#23252B', '#4A4036', '#5B6670', '#3C3C46'];
const SHOE = '#1D1D20';
const BIKE = ['#C8372D', '#2E5FA3', '#2B2F36', '#3E8A5B', '#E3B23C'];
const HELMET = ['#E8E8E4', '#C8372D', '#2E5FA3', '#E3B23C', '#2B2F36'];

const pick = <T>(list: T[], h: number, shift: number): T => list[(h >>> shift) % list.length];

/** A capsule running from `a` to `b` (radius `r`), baked into world-of-the-mesh coordinates. */
function limb(a: THREE.Vector3, b: THREE.Vector3, r: number, hex: string, emit = 0): THREE.BufferGeometry {
  const dir = new THREE.Vector3().subVectors(b, a);
  const len = Math.max(0.001, dir.length());
  const g = new THREE.CapsuleGeometry(r, Math.max(0.001, len - 2 * r), 4, 8);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  const mid = new THREE.Vector3().addVectors(a, b).multiplyScalar(0.5);
  g.applyMatrix4(new THREE.Matrix4().compose(mid, q, new THREE.Vector3(1, 1, 1)));
  return paint(g, hex, emit);
}

const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);

/** A figure whose limbs move; `animate` is driven by distance walked. */
export interface Figure {
  readonly group: THREE.Group;
  /** `phase` is radians of gait/crank cycle; `speed` (m/s) scales the swing. */
  animate(phase: number, speed: number): void;
  dispose(): void;
}

/* ------------------------------------------------------------------ */
/* Pedestrian                                                          */
/* ------------------------------------------------------------------ */

export const PEDESTRIAN_REF = { length: 0.6, width: 0.6, height: 1.75 };

export function buildPedestrian(id: string, material: THREE.Material): Figure {
  const h = hashId(id);
  const skin = pick(SKIN, h, 0);
  const hair = pick(HAIR, h, 3);
  const shirt = pick(SHIRT, h, 6);
  const pants = pick(PANTS, h, 10);

  const group = new THREE.Group();
  const geos: THREE.BufferGeometry[] = [];
  const add = (geo: THREE.BufferGeometry, x = 0, y = 0, z = 0): THREE.Mesh => {
    geos.push(geo);
    const m = new THREE.Mesh(geo, material);
    m.position.set(x, y, z);
    m.castShadow = true;
    m.receiveShadow = true;
    group.add(m);
    return m;
  };

  // Torso, neck, head and hair: one static buffer.
  const body = mergeParts([
    paint(boxAt(0.24, 0.58, 0.42, 0, 1.2, 0), shirt),
    paint(boxAt(0.22, 0.16, 0.36, 0, 0.9, 0), pants),
    paint(new THREE.CylinderGeometry(0.045, 0.05, 0.1, 8).translate(0, 1.53, 0), skin),
    paint(new THREE.SphereGeometry(0.11, 12, 10).translate(0.005, 1.64, 0), skin),
    paint(new THREE.SphereGeometry(0.114, 12, 10).translate(-0.018, 1.655, 0), hair),
  ]);
  add(body);

  // Limbs hang from their joint, so swinging them is a rotation about Z.
  const legGeo = () =>
    mergeParts([
      paint(new THREE.CapsuleGeometry(0.075, 0.74, 4, 8).translate(0, -0.45, 0), pants),
      paint(boxAt(0.2, 0.07, 0.12, 0.05, -0.9, 0), SHOE),
    ]);
  const armGeo = () =>
    mergeParts([
      paint(new THREE.CapsuleGeometry(0.048, 0.34, 4, 8).translate(0, -0.25, 0), shirt),
      paint(new THREE.SphereGeometry(0.045, 8, 6).translate(0, -0.52, 0), skin),
    ]);
  const legL = add(legGeo(), 0, 0.93, 0.1);
  const legR = add(legGeo(), 0, 0.93, -0.1);
  const armL = add(armGeo(), 0, 1.45, 0.25);
  const armR = add(armGeo(), 0, 1.45, -0.25);

  return {
    group,
    animate(phase, speed) {
      const amp = 0.6 * Math.min(1, speed / 1.2);
      const s = Math.sin(phase) * amp;
      legL.rotation.z = s;
      legR.rotation.z = -s;
      armL.rotation.z = -s * 0.8;
      armR.rotation.z = s * 0.8;
      // A small bob keeps the silhouette from looking rigid.
      group.children[0].position.y = Math.abs(Math.cos(phase)) * 0.015 * amp;
    },
    dispose() {
      for (const g of geos) g.dispose();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Rider (shared by the bicycle and the motorcycle)                    */
/* ------------------------------------------------------------------ */

interface RiderPose {
  hip: THREE.Vector3;
  shoulder: THREE.Vector3;
  head: THREE.Vector3;
  grip: THREE.Vector3;
}

/** Static upper body: torso, arms to the grips, head and helmet. */
function riderUpper(
  pose: RiderPose,
  jacket: string,
  skin: string,
  helmet: string,
  fullFace: boolean,
): THREE.BufferGeometry[] {
  const parts: THREE.BufferGeometry[] = [];
  const torso = limb(pose.hip, pose.shoulder, 0.14, jacket);
  torso.scale(1, 1, 1.35);
  parts.push(torso);
  for (const sz of [1, -1]) {
    const sh = pose.shoulder.clone().add(v(0, 0, sz * 0.2));
    const gr = pose.grip.clone().add(v(0, 0, sz * 0.2));
    parts.push(limb(sh, gr, 0.045, jacket));
  }
  const hc = pose.head;
  parts.push(paint(new THREE.SphereGeometry(0.1, 10, 8).translate(hc.x, hc.y, hc.z), skin));
  // Helmet: a slightly larger sphere cap over the head.
  const cap = new THREE.SphereGeometry(fullFace ? 0.135 : 0.125, 12, 8, 0, Math.PI * 2, 0, fullFace ? Math.PI * 0.8 : Math.PI * 0.55);
  cap.translate(hc.x - 0.01, hc.y + 0.02, hc.z);
  parts.push(paint(cap, helmet));
  return parts;
}

/* ------------------------------------------------------------------ */
/* Cyclist                                                             */
/* ------------------------------------------------------------------ */

export const CYCLIST_REF = { length: 1.8, width: 0.7, height: 1.7 };

const CRANK = v(0.0, 0.3, 0);
const CRANK_R = 0.17;
const HIP = v(-0.2, 0.98, 0);

export function buildCyclist(id: string, material: THREE.Material): Figure {
  const h = hashId(id);
  const skin = pick(SKIN, h, 0);
  const jacket = pick(SHIRT, h, 6);
  const pants = pick(PANTS, h, 10);
  const frame = pick(BIKE, h, 14);
  const helmet = pick(HELMET, h, 17);

  const group = new THREE.Group();
  const geos: THREE.BufferGeometry[] = [];
  const addStatic = (parts: THREE.BufferGeometry[]): void => {
    const geo = mergeParts(parts);
    geos.push(geo);
    const m = new THREE.Mesh(geo, material);
    m.castShadow = true;
    m.receiveShadow = true;
    group.add(m);
  };

  const rear = v(-0.52, 0.34, 0);
  const front = v(0.52, 0.34, 0);
  const seat = v(-0.2, 0.9, 0);
  const head = v(0.4, 0.88, 0);
  const bars = v(0.36, 1.03, 0);
  const tube = (a: THREE.Vector3, b: THREE.Vector3, r = 0.022, hex = frame) => limb(a, b, r, hex);

  const bike: THREE.BufferGeometry[] = [];
  for (const w of [rear, front]) {
    bike.push(paint(new THREE.TorusGeometry(0.34, 0.03, 6, 22).translate(w.x, w.y, 0), '#1B1B1E'));
    bike.push(paint(new THREE.CylinderGeometry(0.03, 0.03, 0.06, 8).rotateX(Math.PI / 2).translate(w.x, w.y, 0), '#BFC3C8'));
  }
  bike.push(tube(CRANK, seat), tube(CRANK, head), tube(seat, head), tube(seat, rear), tube(CRANK, rear));
  bike.push(tube(head, front), tube(head, bars));
  bike.push(paint(boxAt(0.04, 0.04, 0.46, bars.x, bars.y, 0), '#25262A'));
  bike.push(paint(boxAt(0.2, 0.05, 0.1, seat.x - 0.02, seat.y + 0.03, 0), '#25262A'));
  bike.push(paint(boxAt(0.04, 0.06, 0.04, front.x + 0.03, 0.82, 0), '#EAF6FF', 0.8));
  bike.push(paint(boxAt(0.03, 0.05, 0.05, rear.x - 0.08, 0.7, 0), '#E5282E', 1));
  addStatic([
    ...bike,
    ...riderUpper(
      { hip: HIP, shoulder: v(0.12, 1.4, 0), head: v(0.22, 1.56, 0), grip: v(0.38, 1.03, 0) },
      jacket,
      skin,
      helmet,
      false,
    ),
  ]);

  // Legs pedal: one capsule per leg, restretched hip -> foot every frame.
  const legGeos = [0, 1].map(() => {
    const g = paint(new THREE.CapsuleGeometry(0.07, 0.8, 4, 8), pants);
    geos.push(g);
    return g;
  });
  const legs = legGeos.map((g, i) => {
    const m = new THREE.Mesh(g, material);
    m.castShadow = true;
    group.add(m);
    return { mesh: m, z: i === 0 ? 0.11 : -0.11, flip: i };
  });
  const up = v(0, 1, 0);
  const place = (phase: number): void => {
    for (const leg of legs) {
      const a = phase + leg.flip * Math.PI;
      const foot = v(CRANK.x + CRANK_R * Math.cos(a), CRANK.y + CRANK_R * Math.sin(a), leg.z);
      const hip = v(HIP.x, HIP.y, leg.z);
      const dir = foot.clone().sub(hip);
      const dist = dir.length();
      leg.mesh.position.copy(hip).add(foot).multiplyScalar(0.5);
      leg.mesh.quaternion.setFromUnitVectors(up, dir.normalize());
      leg.mesh.scale.set(1, dist / (0.8 + 0.14), 1);
    }
  };
  place(0);

  return {
    group,
    animate(phase, speed) {
      place(speed > 0.05 ? phase : 0);
    },
    dispose() {
      for (const g of geos) g.dispose();
    },
  };
}

/* ------------------------------------------------------------------ */
/* Motorcycle                                                          */
/* ------------------------------------------------------------------ */

export const MOTORCYCLE_PAINT = ['#1B1D21', '#B8282E', '#27508C', '#C9CCD0', '#E3B23C'];

/** Static motorcycle with rider, one merged buffer (no animation). */
export function buildMotorcycleGeometry(id: string, bodyHex: string): THREE.BufferGeometry {
  const h = hashId(id);
  const skin = pick(SKIN, h, 0);
  const jacket = pick(['#23252B', '#2B2F36', '#8A2E2A', '#2E4E7E'], h, 6);
  const pants = pick(PANTS, h, 10);
  const helmet = pick(HELMET, h, 17);
  const parts: THREE.BufferGeometry[] = [];

  const rear = v(-0.7, 0.32, 0);
  const front = v(0.72, 0.32, 0);
  for (const w of [rear, front]) {
    parts.push(paint(wheelAt(0.32, 0.14, w.x, w.y, 0), '#1B1B1E'));
    parts.push(paint(wheelAt(0.16, 0.16, w.x, w.y, 0), '#BFC3C8'));
  }
  // Frame, engine, tank, seat, fenders.
  parts.push(paint(boxAt(0.5, 0.3, 0.26, -0.05, 0.42, 0), '#2A2C31'));
  parts.push(paint(boxAt(0.5, 0.2, 0.3, 0.1, 0.78, 0), bodyHex));
  parts.push(paint(boxAt(0.55, 0.1, 0.3, -0.35, 0.78, 0), '#1E1F23'));
  parts.push(paint(boxAt(0.35, 0.05, 0.16, -0.82, 0.62, 0), bodyHex));
  parts.push(limb(v(0.3, 0.8, 0), front, 0.03, '#BFC3C8'));
  parts.push(limb(rear, v(-0.1, 0.45, 0), 0.04, '#2A2C31'));
  parts.push(paint(boxAt(0.05, 0.05, 0.62, 0.34, 1.0, 0), '#25262A'));
  parts.push(paint(boxAt(0.08, 0.12, 0.14, 0.58, 0.86, 0), '#F4FAFF', 0.9));
  parts.push(paint(boxAt(0.04, 0.07, 0.12, -0.92, 0.7, 0), '#FF2A2A', 1));
  parts.push(paint(boxAt(0.02, 0.1, 0.18, -0.93, 0.56, 0), '#EDEBE0'));
  // Rider: seated, leaning to the bars, knees against the tank.
  parts.push(
    ...riderUpper(
      { hip: v(-0.38, 0.88, 0), shoulder: v(-0.02, 1.28, 0), head: v(0.08, 1.45, 0), grip: v(0.34, 1.0, 0) },
      jacket,
      skin,
      helmet,
      true,
    ),
  );
  for (const sz of [1, -1]) {
    parts.push(limb(v(-0.38, 0.88, sz * 0.12), v(0.12, 0.72, sz * 0.17), 0.075, pants));
    parts.push(limb(v(0.12, 0.72, sz * 0.17), v(0.04, 0.34, sz * 0.19), 0.06, pants));
  }
  return mergeParts(parts);
}
